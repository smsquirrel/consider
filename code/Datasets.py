import os
import torch
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
LANGS_TOKENS = ['[Go]', '[Java]', '[JavaScript]', '[PHP]', '[Python]', '[Ruby]']  

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

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

# remove comments, tokenize code and extract dataflow                                        
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


        
        
def convert_examples_to_features(item, lang=None, LT=False, NL_LT=False):
    js,tokenizer,args=item
    MultiLT = args.MultiLT
    #code
    if lang is None:
        lang = args.lang

    if args.model_type=="graphcodebert":
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
        elif MultiLT:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-8-min(len(dfg),args.data_flow_length)]
            code_tokens =[tokenizer.cls_token]+LANGS_TOKENS+code_tokens+[tokenizer.sep_token]
        else:
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg),args.data_flow_length)]
            code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]

        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        if MultiLT:
            position_idx = [tokenizer.pad_token_id + 1 for i in range(6)]
            position_idx += [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens)-6)]   # pad_token_id is 1, start from 2
        else:
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
        elif MultiLT:
            length=len([tokenizer.cls_token] + LANGS_TOKENS)
        else:
            length = len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    elif args.model_type=="codebert" or "roberta" in args.model_type:
        if args.tokenizer_type == "dfg":
            parser=parsers[lang]
            code_tokens,_=extract_dataflow(js['original_string'],parser, lang)
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
            ori2cur_pos={}
            ori2cur_pos[-1]=(0,0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
            code_tokens=[y for x in code_tokens for y in x] 
        else:
            code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
            code_tokens = tokenizer.tokenize(code)
        if LT:
            code_tokens = code_tokens[:args.code_length-3]
            # code_tokens =[tokenizer.cls_token]+[LANGS_TOKENS[L2I[lang]]]+code_tokens+[tokenizer.sep_token]
            code_tokens =[tokenizer.cls_token]+['[LS]']+code_tokens+[tokenizer.sep_token]
        elif MultiLT:
            code_tokens = code_tokens[:args.code_length-8]
            code_tokens =[tokenizer.cls_token]+LANGS_TOKENS+code_tokens+[tokenizer.sep_token]
        else:
            code_tokens = code_tokens[:args.code_length-2]
            code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        # add padding
        padding_length = args.code_length - len(code_tokens)
        code_tokens += [tokenizer.pad_token]*padding_length
        code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
        if MultiLT:
            position_idx = [tokenizer.pad_token_id + 1 for i in range(6)]
            position_idx += [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens)-6)]   # pad_token_id is 1, start from 2
        else:
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(args.code_length)]    
        dfg_to_code, dfg_to_dfg = None, None    
    else:
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        if LT:
            code_tokens = tokenizer.tokenize(code)[:args.code_length-5]
            code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+['[LS]']+code_tokens+[tokenizer.sep_token]
        elif MultiLT:
            code_tokens = tokenizer.tokenize(code)[:args.code_length-10]
            code_tokens =[tokenizer.cls_token]+LANGS_TOKENS+["<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        else:
            code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
            code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id]*padding_length
        if MultiLT:
            position_idx = [tokenizer.pad_token_id + 1 for i in range(8)]
            position_idx += [i+tokenizer.pad_token_id + 1 for i in range(len(code_ids)-8)]   # pad_token_id is 1, start from 2
        else:
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(args.code_length)]
        dfg_to_code, dfg_to_dfg = None, None
        
  
    #nl
    if args.model_type=="unixcoder":
        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        if NL_LT:
            nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-5]
            nl_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+['[LS]']+nl_tokens+[tokenizer.sep_token]
        else:
            nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
            nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id]*padding_length  
    else:
        nl=' '.join(js['docstring_tokens'])
        if NL_LT:
            nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-3]
            nl_tokens =[tokenizer.cls_token]+['[LS]']+nl_tokens+[tokenizer.sep_token]
        else:
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
    def __init__(self, lang, LT=False, NL_LT=False, model_type="graphcodebert"):
        self.lang = lang
        self.LT = LT
        self.NL_LT = NL_LT
        self.model_type = model_type

    def __call__(self, item):
        return convert_examples_to_features(item, self.lang, self.LT, self.NL_LT)
    

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None, ML=False, NL_LT=False):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        self.lang_len = {}

        if args.tokenizer_type != "dfg":
            prefix = prefix + "_" + 'normal'
        if NL_LT:
            prefix = prefix + "_NLLT"
        if args.code_length != 256:
            prefix = prefix + "_cl" + str(args.code_length)
        if ML:
            if args.model_type != "graphcodebert":
                prefix = prefix + "_" + args.model_type            
            if not args.LT:
                if args.MultiLT:
                    prefix = prefix + "_MultiLT"
                else:
                    prefix = prefix + "_woLT"
            
            cache_file = '/'.join(file_path.split('/')[:-1]) + '/' + prefix + '.pkl'
        else:
            cache_file = '/'.join(file_path.split('/')[:-1]) + '/' + prefix + '.pkl'
        logger.info("cache_file: {}".format(cache_file))
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            self.lang_len = {'go':167288, 'java':164923, 'javascript':58025, 'php':241241, 'python':251820, 'ruby':24927}
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
                    file_examples = pool.map(convert_examples_to_features_with_lang(lang, LT=args.LT, model_type=args.model_type), tqdm(data,total=len(data)))
                    self.lang_len[lang] = len(file_examples)
                    self.examples.extend(file_examples)
            elif 'ML' in cache_file and ('codebase' in cache_file or 'valid' in cache_file):
                for lang in LANGS:
                    data=[]
                    if 'codebase' in cache_file:
                        path = './dataset/{}/codebase.jsonl'.format(lang)
                    else:
                        path = './dataset/{}/valid.jsonl'.format(lang)
                    with open(path) as f:
                        for line in f:
                            line=line.strip()
                            js=json.loads(line)
                            data.append((js,tokenizer,args))
                    file_examples = pool.map(convert_examples_to_features_with_lang(lang, LT=args.LT, model_type=args.model_type), tqdm(data,total=len(data)))
                    self.lang_len[lang] = len(file_examples)
                    self.examples.extend(file_examples)   
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
                self.examples=pool.map(convert_examples_to_features_with_lang(lang, LT=args.LT, model_type=args.model_type), tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))

        if 'train' in file_path:
            self.Mode = 'train'
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))     
        else:
            self.Mode = 'test'     
                
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
            # add all language special tokens attention
            for idx,i in enumerate(self.examples[item].position_idx):
                if i in [2]:
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
                        attn_mask[idx+node_index,a+node_index]=True   # 为何使用的单向的边

        if "graphcodebert" in self.args.model_name_or_path:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids), 
                torch.tensor(self.examples[item].lang))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(self.examples[item].position_idx), 
                torch.tensor(self.examples[item].nl_ids),
                torch.tensor(self.examples[item].lang))


class SamplerIterator():
    def __init__(self, examples, args, num_batches, batchsize=32, alpha=0.5, beta=1, mode="loader"):
        self.args = args
        self.batchsize = batchsize
        self.alpha = alpha
        self.beta = beta
        self.num_batches = num_batches
        self.LANGS = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
        self.lang_datasets = {'go': [], 'java': [], 'javascript': [], 'php': [], 'python': [], 'ruby': []}
        self.lang_P = {}
        self.lang_nums = {'go': 0, 'java': 0, 'javascript': 0, 'php': 0, 'python': 0, 'ruby': 0}
        for example in examples:
            if mode == "loader":
                lang = LANGS[example[-1]]
            else:
                lang = LANGS[example.lang]
            self.lang_datasets[lang].append(example)
            self.lang_nums[lang] += 1
        for lang in LANGS:
            self.lang_P[lang] = np.array([1.0]*self.lang_nums[lang])
        self.lang_class_P = np.array([len(self.lang_datasets[lang]) for lang in LANGS])
        self.lang_class_P = self.lang_class_P / sum(self.lang_class_P)
        # self.lang_class_P = dict(zip(LANGS, self.lang_class_P))

        self.ConfusionMatrix = np.eye(len(self.LANGS)) + 1   
        self.real_nums = np.zeros(len(self.LANGS))


    def __iter__(self):
        for _ in range(self.num_batches):
            real_P = self.real_nums / self.real_nums.sum()
            self.lang_class_P = self.lang_class_P * (self.lang_class_P/real_P)
            self.lang_class_P = self.lang_class_P / self.lang_class_P.sum()
            selected_lang = np.random.choice(np.arange(6), size=1, p=self.lang_class_P).squeeze()
            origin_P = self.ConfusionMatrix[selected_lang,:]
            origin_P = origin_P / origin_P.sum() 
            if origin_P[selected_lang] < self.alpha:
                fusion_P += (self.alpha - origin_P[selected_lang]) ** self.beta 
            fusion_P = fusion_P / fusion_P.sum()
            load_lang_nums = np.round(self.batchsize * fusion_P).astype(int)
            load_lang_nums[-1] = self.batchsize - sum(load_lang_nums[:-1])
            self.real_nums = self.real_nums + load_lang_nums
            load_lang_nums = dict(zip(LANGS, load_lang_nums.tolist()))
            code_ids_batch, attn_mask_batch, position_idx_batch, nl_ids_batch, lang_batch = [], [], [], [], []
            for lang, nums in load_lang_nums.items():
                if nums > 0:
                    lang_dataset = self.lang_datasets[lang]
                    choiceIdx = np.random.choice(np.arange(self.lang_nums[lang]), size=nums, p=self.lang_P[lang]/sum(self.lang_P[lang]), replace=False)
                    for idx in choiceIdx:
                        code_ids_batch.append(lang_dataset[idx][0])
                        position_idx_batch.append(lang_dataset[idx][-3])
                        nl_ids_batch.append(lang_dataset[idx][-2])
                        lang_batch.append(lang_dataset[idx][-1])
                        if "graphcodebert" in self.args.model_name_or_path:
                            attn_mask_batch.append(lang_dataset[idx][1])
                    self.lang_P[lang][choiceIdx] = self.lang_P[lang][choiceIdx] / 2
                    # code_ids_batch = [torch.tensor(item.code_ids) for item in sampler_data]
                    # position_idx_batch = [torch.tensor(item.position_idx) for item in sampler_data]
                    # nl_ids_batch = [torch.tensor(item.nl_ids) for item in sampler_data]
                    # lang_batch = [torch.tensor(item.lang) for item in sampler_data]
                    # sample_idx = np.random.choice(np.arange(len(lang_dataset)), size=nums, p=lang_P[lang]/sum(lang_P[lang]), replace=False)
                    # for idx in sample_idx:
                    #     sampler_data.append(lang_dataset[idx]) 
                    #     lang_P[lang][idx] = lang_P[lang][idx] / 2

            code_ids_batch, position_idx_batch, nl_ids_batch, lang_batch = torch.stack(code_ids_batch), torch.stack(position_idx_batch), torch.stack(nl_ids_batch), torch.stack(lang_batch)
            if "graphcodebert" in self.args.model_name_or_path:
                attn_mask_batch = torch.stack(attn_mask_batch)
                yield code_ids_batch, attn_mask_batch, position_idx_batch, nl_ids_batch, lang_batch
            else:
                yield code_ids_batch, position_idx_batch, nl_ids_batch, lang_batch

class SamplerIterableDataset(IterableDataset):
    def __init__(self, iterator):
        super().__init__()
        self.iterator = iterator

    def __iter__(self):
        return iter(self.iterator)
