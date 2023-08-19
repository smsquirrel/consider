import argparse
import logging
import os
import pickle
import random
import torch
import json
import time
import shutil
import numpy as np
from model import Model, DomainClassifier, CodeSummarization, PreCodeRetrieval, CRNet
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, BatchSampler, TensorDataset, ConcatDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm
import multiprocessing
cpu_cont = 16

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}
dataset_path = './dataset'
LANGS = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
L2I = {'go': 0, 'java': 1, 'javascript': 2,
        'php': 3, 'python': 4, 'ruby': 5}
LANGS_TOKENS = ['<go>', '<java>', '<javascript>', '<php>', '<python>', '<ruby>']  # TODO: alter 括号, 更改在输入中的位置
# LANGS = ['python', 'ruby']
temperture = 1


#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
        # ALTER: TODO: analyze nums of failed cases
        code_tokens = []
    return code_tokens,dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,
                 lang,
                 code_teacher_emb=None,
                 nl_teacher_emb=None,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.lang=lang
        self.code_teacher_emb = code_teacher_emb 
        self.nl_teacher_emb = nl_teacher_emb
        
        
def convert_examples_to_features(item, lang=None, LT=False, model_type="graphcodebert"):
    js,tokenizer,args=item
    #code
    if lang is None:
        lang = args.lang

    if model_type=="graphcodebert":
        parser=parsers[lang]
        #extract data flow
        code_tokens,dfg=extract_dataflow(js['original_string'],parser, lang)
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        #truncating
        if LT:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)]
            # code_tokens =[tokenizer.cls_token]+[LANGS_TOKENS[L2I[lang]]]+code_tokens+[tokenizer.sep_token]
            code_tokens =[tokenizer.cls_token]+['[LS]']+code_tokens+[tokenizer.sep_token]
        else:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
            code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]

        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]   # pad_token_id is 1, start from 2
        dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
        code_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]   # dfg token position is 0
        code_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.code_length+args.data_flow_length-len(code_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length  # extra postion id is 1
        code_ids+=[tokenizer.pad_token_id]*padding_length    
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        if LT:
            length=len([tokenizer.cls_token] + [LANGS_TOKENS[L2I[lang]]])  
        else:
            length = len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    elif model_type=="codebert" or "roberta" in model_type:
        code_tokens = js['original_string']
        code_tokens = tokenizer.tokenize(code_tokens)
        code_tokens = code_tokens[:args.code_length-2]
        code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        # add padding
        padding_length = args.code_length - len(code_tokens)
        code_tokens += [tokenizer.pad_token]*padding_length
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(args.code_length)]    
        dfg_to_code, dfg_to_dfg = None, None    
    else:
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
        code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
        position_idx, dfg_to_code, dfg_to_dfg = None, None, None  
        
  
    #nl
    if model_type=="unixcoder":
        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
        nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length  
    else:
        nl=' '.join(js['docstring_tokens'])
        nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
        nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids+=[tokenizer.pad_token_id]*padding_length    

    # lang
    if lang is not None:
        lang = L2I[lang]
    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'],lang)


class convert_examples_to_features_with_lang:
    def __init__(self, lang, LT=False, model_type="graphcodebert"):
        self.lang = lang
        self.LT = LT
        self.model_type = model_type

    def __call__(self, item):
        return convert_examples_to_features(item, self.lang, self.LT, self.model_type)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None, ML=False):
        self.args=args
        self.model_mode = 'teacher'
        prefix=file_path.split('/')[-1][:-6]  # dataset/{}/train.jsonl
        self.lang_len = {}
        if "graphcodebert" in args.model_name_or_path:
            model_type = "graphcodebert"
        elif "codebert" in args.model_name_or_path:
            model_type = "codebert"
        else:
            model_type = "unixcoder"
        # cache_file=args.output_dir+'/'+prefix+'.pkl'
        if args.code_length != 256:
            prefix = prefix + "_cl" + str(args.code_length)
        if model_type != "graphcodebert":
            prefix = prefix + "_" + model_type
        if ML:
            if not args.LT:
                prefix = prefix + "_woLT"
            cache_file = '/'.join(file_path.split('/')[:-1]) + '/' + prefix + '.pkl'
        else:
            cache_file = '/'.join(file_path.split('/')[:-1]) + '/' + prefix + '.pkl'
        logger.info("load cache_file: {}".format(cache_file))
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            self.lang_len = {'go':167288, 'java':164923, 'javascript':58025, 'php':241241, 'python':251820, 'ruby':24927}
            # for lang in LANGS:
            #     path = "./dataset/{}/train.jsonl".format(lang)
            #     with open(path) as f:
            #         self.lang_len[lang] = len(f.readlines())
        else:
            self.examples = []

            if ML and 'train' in file_path:
                for lang in LANGS:
                    data=[]
                    path = "./dataset/{}/train.jsonl".format(lang)
                    with open(path) as f:
                        for line in f:
                            line=line.strip()
                            js=json.loads(line)
                            data.append((js,tokenizer,args))
                    file_examples = pool.map(convert_examples_to_features_with_lang(lang, LT=args.LT, model_type=model_type), tqdm(data,total=len(data)))
                    # logger.info(f"data length: {len(data)}, file_examples length: {len(file_examples)}")
                    self.lang_len[lang] = len(file_examples)
                    self.examples.extend(file_examples)
                # logger.info(f"lang len: {self.lang_len}, total len: {len(self.examples)}")
            else:
                lang = None
                data = []
                for alang in LANGS:
                    if alang in file_path:
                        lang = alang
                        break 
                if 'javascript' in file_path:
                    lang = 'javascript'
                logger.info("building dataset, file_path: {} lang: {}".format(file_path, lang))
                with open(file_path) as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
                self.examples=pool.map(convert_examples_to_features_with_lang(lang, LT=args.LT, model_type=model_type), tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
        # self.examples = self.examples[:4000]

        if 'train' in file_path:
            self.Mode = 'train'
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                # logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                # logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))     
        else:
            self.Mode = 'test'     
                
    def __len__(self):
        return len(self.examples)

    def load_teacher_emb(self, code_teacher_embs, nl_teacher_embs):
        for i in tqdm(range(len(self.examples)), desc="load teacher emb..."):
            exmaple = self.examples[i]
            item = InputFeatures(exmaple.code_tokens,exmaple.code_ids,exmaple.position_idx,exmaple.dfg_to_code,exmaple.dfg_to_dfg,exmaple.nl_tokens,exmaple.nl_ids,exmaple.url,exmaple.lang, code_teacher_embs[i], nl_teacher_embs[i])
            self.examples[i] = item

    def checkoutMode(self):
        if self.model_mode == 'teacher':
            self.model_mode = "student"
        else:
            self.model_mode = "teacher"

    def __getitem__(self, item): 
        if "graphcodebert" in self.args.model_name_or_path:
            #calculate graph-guided masked function
            attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length),dtype=bool)
            #calculate begin index of node and max length of input
            node_index=sum([i>1 for i in self.examples[item].position_idx])   # code length
            max_length=sum([i!=1 for i in self.examples[item].position_idx])  # code + dfg length
            #sequence can attend to sequence
            attn_mask[:node_index,:node_index]=True
            #special tokens attend to all tokens
            for idx,i in enumerate(self.examples[item].code_ids):
                if i in [0,2]:
                    attn_mask[idx,:max_length]=True
            #nodes attend to code tokens that are identified from
            for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
                if a<node_index and b<node_index:
                    attn_mask[idx+node_index,a:b]=True   # dfg to code edge
                    attn_mask[a:b,idx+node_index]=True   # code to dfg edge
            #nodes attend to adjacent nodes 
            for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(self.examples[item].position_idx):
                        attn_mask[idx+node_index,a+node_index]=True   

        if self.Mode == 'train':
            if self.model_mode == 'teacher':
                return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(attn_mask),
                    torch.tensor(self.examples[item].position_idx), 
                    torch.tensor(self.examples[item].nl_ids), 
                    torch.tensor(self.examples[item].lang))
            else:
                return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(attn_mask),
                    torch.tensor(self.examples[item].position_idx), 
                    torch.tensor(self.examples[item].nl_ids),
                    torch.tensor(self.examples[item].lang),
                    torch.tensor(self.examples[item].code_teacher_emb),
                    torch.tensor(self.examples[item].nl_teacher_emb))


        if "graphcodebert" in self.args.model_name_or_path:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].nl_ids))
            
class StudentDataset(Dataset):
    def __init__(self, args, TeacherDatasets, Mode='train'):
        self.args = args
        self.Mode = Mode
        self.examples = []
        for dataset in TeacherDatasets:
            self.examples.extend(dataset.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        if "graphcodebert" in self.args.model_name_or_path:
            #calculate graph-guided masked function
            attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length),dtype=bool)
            #calculate begin index of node and max length of input
            node_index=sum([i>1 for i in self.examples[item].position_idx])   # code length
            max_length=sum([i!=1 for i in self.examples[item].position_idx])  # code + dfg length
            #sequence can attend to sequence
            attn_mask[:node_index,:node_index]=True
            #special tokens attend to all tokens
            for idx,i in enumerate(self.examples[item].code_ids):
                if i in [0,2]:
                    attn_mask[idx,:max_length]=True
            #nodes attend to code tokens that are identified from
            for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
                if a<node_index and b<node_index:
                    attn_mask[idx+node_index,a:b]=True   # dfg to code edge
                    attn_mask[a:b,idx+node_index]=True   # code to dfg edge
            #nodes attend to adjacent nodes 
            for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(self.examples[item].position_idx):
                        attn_mask[idx+node_index,a+node_index]=True   

        if "graphcodebert" in self.args.model_name_or_path:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids), 
                torch.tensor(self.examples[item].lang),
                self.examples[item].code_teacher_emb,
                self.examples[item].nl_teacher_emb)
        else:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].lang),
                self.examples[item].code_teacher_emb,
                self.examples[item].nl_teacher_emb)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer,pool, ML=False, DCmodel=None, CRModel=None):
    """ Train the model """
    # step1: contruct teacher dataset and teacher model 
    # step2: use teacher model to eval teacher dataset to get code_teacher_emb and nl_teacher_emb
    # step3: use code_teacher_emb and nl_teacher_emb to train student model
    teacher_datasets = [TextDataset(tokenizer, args, 'dataset/{}/train.jsonl'.format(lang), pool, ML=False) for lang in LANGS]
    teacher_dataloaders = []
    for dataset in teacher_datasets:
        teacher_dataloaders.append(DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8))
    
    for i in range(len(LANGS)):
        lang = LANGS[i]
        logger.info('{} teacher embedding generating...'.format(lang))
        teacher_emb_path = 'dataset/teacher_emb/codebert_{}.pkl'.format(lang)
        if os.path.exists(teacher_emb_path):
            with open(teacher_emb_path, 'rb') as f:
                code_teacher_embs, nl_teacher_embs = pickle.load(f)
                teacher_datasets[i].load_teacher_emb(code_teacher_embs, nl_teacher_embs)
            continue
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        tokenizer.add_tokens(LANGS_TOKENS)
        tokenizer.add_tokens(['[LS]'])
        teacher_model = RobertaModel.from_pretrained("microsoft/codebert-base")    
        teacher_model.resize_token_embeddings(len(tokenizer))
        teacher_model = Model(teacher_model, args)
        teacher_model_path = 'saved_models/teacher_model/codebert_{}.bin'.format(lang)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=args.device),strict=False)      
        teacher_model.to(args.device)
        teacher_model.eval()
        code_vecs, nl_vecs = [], []
        for step, batch in tqdm(enumerate(teacher_dataloaders[i]), desc="teacher eval..."):
            # get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            lang_labels = batch[4].to(args.device)
            # forward
            with torch.no_grad():
                nl_vec = teacher_model(nl_inputs=nl_inputs)[1]
                code_vec = teacher_model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)[1]
            code_vecs.append(code_vec.cpu())
            nl_vecs.append(nl_vec.cpu())
        code_vecs = torch.cat(code_vecs, dim=0)
        nl_vecs = torch.cat(nl_vecs, dim=0)
        # save teacher emb to pkl
        with open(teacher_emb_path, 'wb') as f:
            pickle.dump((code_vecs, nl_vecs), f)
        teacher_datasets[i].load_teacher_emb(code_vecs, nl_vecs)
    student_datasets = StudentDataset(args, teacher_datasets)
    student_dataloader = DataLoader(student_datasets, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
   
    #get optimizer and scheduler
    if args.Summarization:
        CRoptim = AdamW(CRModel.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(CRoptim, num_warmup_steps=0,num_training_steps=len(student_dataloader)*args.num_train_epochs)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(student_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.Summarization:
            CRModel = torch.nn.DataParallel(CRModel)
        else:
            model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(student_datasets))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(student_dataloader)*args.num_train_epochs)
    
    if args.Summarization:
        CRModel.zero_grad()
        CRModel.train()
    else:
        model.zero_grad()
        model.train()
    scaler = GradScaler()
    tr_num, tr_loss, best_mrr, train_dc_loss, train_mdc_loss, train_cs_loss, li_loss = 0, 0, 0, 0, 0, 0, 0
    for idx in range(args.num_train_epochs): 
        
        for step, batch in enumerate(student_dataloader):
            # get inputs
            code_inputs = batch[0].to(args.device)  
            if "graphcodebert" in args.model_name_or_path:
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                nl_inputs = batch[3].to(args.device)
                lang_labels = batch[4].to(args.device)
                code_teacher_emb = batch[5].to(args.device)
                nl_teacher_emb = batch[6].to(args.device)
            else:
                nl_inputs = batch[1].to(args.device)
                lang_labels = batch[2].to(args.device)
                code_teacher_emb = batch[-2].to(args.device)
                nl_teacher_emb = batch[-1].to(args.device)
                attn_mask, position_idx = None, None
            with autocast():
                nl_outputs = model(nl_inputs=nl_inputs)
                nl_vec = nl_outputs[1]  

                code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)[1]
                teacher_code_with_student_nl = torch.einsum("ab,cb->ac",nl_vec,code_teacher_emb)
                teacher_nl_with_student_code = torch.einsum("ab,cb->ac",nl_teacher_emb,code_vec)
                scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
                # scores = scores + teacher_code_with_student_nl + teacher_nl_with_student_code
                
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(scores/temperture, torch.arange(code_inputs.size(0), device=scores.device))
                loss += loss_fct(teacher_code_with_student_nl/temperture, torch.arange(code_inputs.size(0), device=scores.device))
                loss += loss_fct(teacher_nl_with_student_code/temperture, torch.arange(code_inputs.size(0), device=scores.device))
                tr_loss += loss.item()

            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {} li_loss: {} cs_loss: {} dc_loss: {} mdc_loss: {} ".format(idx, step+1, round(tr_loss/tr_num,5), round(li_loss/tr_num,5), round(train_cs_loss/tr_num, 5), round(train_dc_loss/tr_num, 5), round(train_mdc_loss/tr_num, 5)))
                tr_loss, train_cs_loss, train_dc_loss, train_mdc_loss, li_loss = 0, 0, 0, 0, 0                
                tr_num=0
            
            #backward
            scaler.scale(loss).backward()
            if args.Summarization:
                scaler.step(CRoptim)
            else:
                scaler.step(optimizer)
            scaler.update()

            if args.Summarization:
                CRoptim.zero_grad()
            else:  
                optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        if ML:
            results = {"eval_mrr":0}
            for lang in LANGS:
                path = 'dataset/{}/valid.jsonl'.format(lang)
                result = evaluate(args, model, tokenizer, path, pool, eval_when_training=True, CRModel=CRModel)
                results['eval_mrr']+=result['eval_mrr']
                results[lang] = result['eval_mrr']
            results['eval_mrr']/=len(LANGS)
        else:
            results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True, CRModel=CRModel)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr-codebert-distill'    # checkpoint-best-mrr-BS70-tempAnneal-LT-CN -BS240-maxLTLI-norm
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)             
            if args.Summarization:
                model_to_save = CRModel.module if hasattr(CRModel,'module') else CRModel
            else:        
                model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False, CRModel=None, evalML=False):
    ML = False
    if args.lang == 'ML':
        ML = True
    logger.info("***** Evaluation file name:{} *****".format(file_name))
    if evalML:
        file_name = 'dataset/ML/valid.jsonl'
    query_dataset = TextDataset(tokenizer, args, file_name, pool, ML=ML)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)

    if ML:
        for alang in LANGS:
            if alang in file_name:
                lang = alang
                break 
        if 'javascript' in file_name:
            lang = 'javascript'
        codebase_file = 'dataset/{}/codebase.jsonl'.format(lang)
    else:
        codebase_file = args.codebase_file
    if evalML:
        codebase_file = 'dataset/ML/codebase.jsonl'
    code_dataset = TextDataset(tokenizer, args, codebase_file, pool, ML=ML)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)  

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        if args.Summarization:
            CRModel = torch.nn.DataParallel(CRModel)
        else:
            model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.Summarization:
        CRModel.eval()
        model = CRModel.model
    model.eval()
    code_vecs, code_embs, lt_embs=[], [], [] 
    nl_vecs=[]
    for batch in query_dataloader:  
        if "graphcodebert" in args.model_name_or_path:
            nl_inputs = batch[3].to(args.device)
        else:
            nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            with autocast():
                nl_vec = model(nl_inputs=nl_inputs)[1]
                nl_vecs.append(nl_vec.detach().cpu()) 
    batch_idx = 0
    for batch in code_dataloader:
        if "graphcodebert" in args.model_name_or_path:
            code_inputs = batch[0].to(args.device)    
            attn_mask = batch[1].to(args.device)
            position_idx =batch[2].to(args.device)
        else:
            code_inputs = batch[0].to(args.device)  
            attn_mask, position_idx = None, None
        with torch.no_grad():
            with autocast():
                CLS_emb, LT_emb = None, None
                if args.Summarization:
                    code_vec = CRModel(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx, nl_inputs=None, nl_length=None, mode="test")  
                    if hasattr(code_vec, 'last_hidden_state'):
                        code_vec = code_vec.last_hidden_state[:, :2, :]
                        code_vec = torch.nn.functional.normalize(code_vec, p=2, dim=2)
                        CLS_emb = code_vec[:, 0, :]
                        LT_emb = code_vec[:, 1, :]
                else:
                    code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)[1]
            if CLS_emb is not None:
                code_embs.append(CLS_emb.detach().cpu())
                lt_embs.append(LT_emb.detach().cpu())
            else:
                code_vecs.append(code_vec.detach().cpu())  
    model.train()    
    if args.Summarization:
        CRModel.train()
    if code_vecs != []:
        code_vecs=torch.cat(code_vecs,0).float()
    else:
        code_embs=torch.cat(code_embs,0).float()
        lt_embs=torch.cat(lt_embs,0).float()
    nl_vecs=torch.cat(nl_vecs,0).float()
    NBS = 64
    CodeLength = len(code_dataset)
    scores = []

    if code_vecs != []:
        for i in range(CodeLength // NBS + 1):  # n, hidden * nbs, seq, hidden = n, nbs, seq
            if i*NBS>=CodeLength:
                break
            if CLS_emb is not None:
                score = torch.max(torch.stack([torch.einsum("ab,cb->ac", nl_vecs.to(args.device), code_embs[i*NBS:(i+1)*NBS,:].to(args.device)), torch.einsum("ab,cb->ac", nl_vecs.to(args.device), lt_embs[i*NBS:(i+1)*NBS,:].to(args.device))], dim=1), dim=1)[0]
            else:
                score = torch.einsum("ab,cb->ac", nl_vecs.to(args.device), code_vecs[i*NBS:(i+1)*NBS,:].to(args.device))
            # score = torch.max(score, dim=2)[0]
            scores.append(score.cpu())
        scores = torch.cat(scores, 1)
    else:
        scores=np.maximum(np.matmul(nl_vecs,code_embs.T), np.matmul(nl_vecs,lt_embs.T))  # b, hidden * hidden, b = b,b
    
    scores = scores.cpu().numpy()
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    # each nl to sort from big to small scores
    

        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument('--DC', action='store_true',
                help="Use domain classifier.")
    parser.add_argument('--DomainClassifier', action='store_true',
                help="Use domain classifier.")
    parser.add_argument('--LT', action='store_true',
                help="Use language tokens.")
    parser.add_argument('--BSampler', action='store_true',
                help="Use Batch sampler.")
    parser.add_argument('--Summarization', action='store_true',
                help="Use code summarization as extra task.")
    parser.add_argument('--ACTION', action='store_true',
                help="Use DFG + ACTION prediction as extra task.")
    parser.add_argument('--CN', action='store_true',
                help="Use Control Nodes as multi vector.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #print arguments
    args = parser.parse_args()
    gpu_num = '0'
    #set device
    device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

    pool = multiprocessing.Pool(cpu_cont)

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    args.n_gpu = 1
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(LANGS_TOKENS)
    model = RobertaModel.from_pretrained(args.model_name_or_path)    
    model.resize_token_embeddings(len(tokenizer))
    model=Model(model, args)

    DCmodel = None
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    if args.Summarization:
        CSModel = None
        PCModel = None
        CRModel = CRNet(args, model=model, PCModel=PCModel, CSModel=CSModel)
        CRModel.to(args.device)
    else:
        CRModel = None

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool, ML=True if args.lang=='ML' else False, DCmodel=DCmodel, CRModel=CRModel)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if args.Summarization:
            CRModel.to(args.device)
            CRModel.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)
        model.to(args.device)
        model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
        
        result=evaluate(args, model, tokenizer, args.eval_data_file, pool, CRModel=CRModel)
         
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if args.Summarization:
            CRModel.to(args.device)
            CRModel.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
            result=evaluate(args, model, tokenizer, args.test_data_file, pool, CRModel=CRModel)
        else:
            model.to(args.device)
            model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
            
            result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()