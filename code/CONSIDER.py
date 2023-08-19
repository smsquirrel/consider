import torch
import torch.nn as nn

import os
import json
import pickle
import random
import numpy as np
import argparse
import logging
from Datasets import TextDataset, SamplerIterator, SamplerIterableDataset, DataLoaderX
from sklearn.metrics import confusion_matrix
from model import Model, CRNet
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, BatchSampler, TensorDataset, IterableDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm
import multiprocessing
cpu_cont = 16

dataset_path = './dataset'
LANGS = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
L2I = {'go': 0, 'java': 1, 'javascript': 2,
        'php': 3, 'python': 4, 'ruby': 5}
LANGS_TOKENS = ['[Go]', '[Java]', '[JavaScript]', '[PHP]', '[Python]', '[Ruby]']  
temperture = 1
rounds = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer, pool, ML=False, DCmodel=None, CRModel=None):
    """ Train the model """
    global rounds
    ExtraData, SamplerData = False, True
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool, ML=ML, NL_LT=args.NL_LT)
    if args.BSampler:
        start_idx = 0
        indice_idx = []
        logger.info(f"train_dataset.lang_len: {train_dataset.lang_len}")
        for lang in LANGS:
            indice_list = random.sample(range(start_idx, start_idx + train_dataset.lang_len[lang]), k=train_dataset.lang_len[lang])
            start_idx += train_dataset.lang_len[lang]
            indice_idx += indice_list
        BS = args.train_batch_size
        LEN = len(train_dataset)
        logger.info(f"LEN: {LEN}, indice_idx length: {len(indice_idx)}")
        seqIdx = [[indice_idx[i*BS + j] for j in range(BS)] for i in range(LEN // BS)]
        random.shuffle(seqIdx)
        seqIdx += [[indice_idx[i] for i in range(LEN // BS * BS, LEN)]]
        randomIdx = [i for ii in seqIdx for i in ii]
        train_sampler = BatchSampler(randomIdx, batch_size=BS, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler,num_workers=4)
    else:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    if args.MCL and SamplerData:
        SI = SamplerIterator(train_dataset, args, num_batches=len(train_dataloader) * 2, batchsize=args.train_batch_size)
        sampler_dataset = SamplerIterableDataset(SI)
        sampler_dataloader = DataLoaderX(sampler_dataset, batch_size=None, pin_memory=True, num_workers=0)
    
    #get optimizer and scheduler
    if args.Commonalities:
        CRoptim = AdamW(CRModel.parameters(), lr=args.learning_rate, eps=1e-8)

        if args.MCL and not ExtraData:
            num_training_steps = len(train_dataloader)*args.num_train_epochs*2
        else:
            num_training_steps = len(train_dataloader)*args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(CRoptim, num_warmup_steps=0,num_training_steps=num_training_steps)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=num_training_steps)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.Commonalities:
            CRModel = torch.nn.DataParallel(CRModel)
        else:
            model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    if args.Commonalities:
        CRModel.zero_grad()
        CRModel.train()
    else:
        model.zero_grad()
        model.train()
    scaler = GradScaler()


    tr_num, tr_loss, best_mrr, train_dc_loss, train_mdc_loss, train_cs_loss, train_length_loss, li_loss = 0, 0, 0, 0, 0, 0, 0, 0
    for idx in range(args.num_train_epochs): 
        if args.MCL:
            start_idx = 0
            indice_idx = []
            for lang in LANGS:
                indice_list = random.sample(range(start_idx, start_idx + train_dataset.lang_len[lang]), k=train_dataset.lang_len[lang])
                start_idx += train_dataset.lang_len[lang]
                indice_idx += indice_list
            BS = args.train_batch_size
            LEN = len(train_dataset)
            seqIdx = [[indice_idx[i*BS + j] for j in range(BS)] for i in range(LEN // BS)]
            random.shuffle(seqIdx)
            seqIdx += [[indice_idx[i] for i in range(LEN // BS * BS, LEN)]]
            randomIdx = [i for ii in seqIdx for i in ii]
            train_sampler = BatchSampler(randomIdx, batch_size=BS, drop_last=False)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
            if args.MCL:
                ml_train_sampler = RandomSampler(train_dataset)
                ml_train_dataloader = DataLoader(train_dataset, sampler=ml_train_sampler, batch_size=args.train_batch_size, num_workers=4)
                ml_train_iter = iter(ml_train_dataloader)
                
        rounds += 1
        for step, batch in enumerate(sampler_dataloader):
            if step > len(train_dataloader) * 2:
                break
            if args.MCL and not ExtraData and not SamplerData:
                step_nums = 2
            else:
                step_nums = 1
            for k in range(step_nums):
                if k == 1:
                    batch = next(ml_train_iter)

                code_inputs = batch[0].to(args.device)  
                if "graphcodebert" in args.model_name_or_path:
                    attn_mask = batch[1].to(args.device)
                    position_idx = batch[2].to(args.device)
                    nl_inputs = batch[3].to(args.device)
                    lang_labels = batch[4].to(args.device)
                else:
                    position_idx = batch[1].to(args.device)
                    nl_inputs = batch[2].to(args.device)
                    lang_labels = batch[3].to(args.device)
                    attn_mask = None
                    
                with autocast():
                    #get code and nl vectors
                    scores, LI_scores = None, None  
                    if args.Commonalities:
                        results = CRModel(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx, nl_inputs=nl_inputs, lang_labels = lang_labels, step=idx, nl_length=None)  
                        query_emb, LI_emb, LS_emb, fusionEmb, adv_loss, cs_loss, length_loss = results["query_emb"], results["LI_emb"], results["LS_emb"], results["fusion_emb"], results["adv_loss"], results["cs_loss"], results["length_loss"]

                        if hasattr(fusionEmb, 'last_hidden_state'):
                            code_vecs = fusionEmb.last_hidden_state[:, :2, :]
                            # normalize
                            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
                            CLS_emb = code_vecs[:, 0, :]
                            LT_emb = code_vecs[:, 1, :]
                            scores=torch.max(torch.stack([torch.einsum("ab,cb->ac",query_emb, CLS_emb), torch.einsum("ab,cb->ac",query_emb, LT_emb)], dim=2), dim=2)[0]    
                        elif fusionEmb is not None:
                            scores = torch.einsum("ab,cb->ac",query_emb, fusionEmb)
                        else:
                            LI_scores = torch.einsum("ab,cb->ac",query_emb, LI_emb)
                        if args.n_gpu > 1:
                            if cs_loss is not None:
                                cs_loss = cs_loss.mean()
                            if adv_loss is not None:
                                adv_loss = adv_loss.mean()
                            if length_loss is not None:
                                length_loss = length_loss.mean()
                    else:
                        nl_outputs = model(nl_inputs=nl_inputs)
                        nl_vec = nl_outputs[1]  
                        code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)[1]
                        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
                    
                    loss_fct = CrossEntropyLoss()
                    loss = 0
                    if scores is not None:
                        loss = loss_fct(scores/temperture, torch.arange(code_inputs.size(0), device=args.device))
                    if LI_scores is not None:
                        LIloss = loss_fct(LI_scores/temperture, torch.arange(code_inputs.size(0), device=args.device))
                        li_loss += LIloss.item()
                    else:
                        LIloss = 0
                    loss += LIloss
                    tr_loss += loss.item()
                    if type(LIloss) != int:
                        tr_loss -= LIloss.item()
                        

                    if args.Commonalities and length_loss is not None:
                        loss += length_loss
                        loss += cs_loss
                        train_length_loss += length_loss.item()
                
                #report loss
                if args.n_gpu > 1:
                    loss = loss.mean()


                if args.Commonalities and cs_loss is not None:
                    train_cs_loss += cs_loss.item()
                    # train_cs_loss += cs_loss.item()
                tr_num+=1
                if (step+1)% 100==0:
                    logger.info("epoch {} step {} loss {} li_loss: {} cs_loss: {} dc_loss: {} mdc_loss: {} train_length_loss: {}".format(idx, step+1, round(tr_loss/tr_num,5), round(li_loss/tr_num,5), round(train_cs_loss/tr_num, 5), round(train_dc_loss/tr_num, 5), round(train_mdc_loss/tr_num, 5), round(train_length_loss/tr_num, 5)))
                    tr_loss, train_cs_loss, train_dc_loss, train_mdc_loss, train_length_loss, li_loss = 0, 0, 0, 0, 0, 0                
                    tr_num=0
                
                #backward
                scaler.scale(loss).backward(retain_graph=True if ExtraData else False)

                if args.Commonalities:
                    scaler.step(CRoptim)
                else:
                    scaler.step(optimizer)
                scaler.update()

                if args.Commonalities:
                    CRoptim.zero_grad()
                else:  
                    optimizer.zero_grad()
                scheduler.step() 
                    
        #evaluate    
        if ML:
            results = {"eval_mrr":0, "lang_distr":[]}
            for lang in LANGS:
                path = 'dataset/{}/valid.jsonl'.format(lang)
                result = evaluate(args, model, tokenizer, path, pool, eval_when_training=True, CRModel=CRModel, evalML=True)
                results['eval_mrr']+=result['eval_mrr']
                results[lang] = result['eval_mrr']
                results['lang_distr'].append(result['lang_distr'])
            results['eval_mrr']/=len(LANGS)
        else:
            results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True, CRModel=CRModel, evalML=True)
        for key, value in results.items():
            if key == "eval_mrr":
                logger.info("  %s = %s", key, round(value,4))
        sampler_dataloader.dataset.ConfusionMatrix = np.array(results["lang_distr"])
        logger.info(f"sampler data distribution: {sampler_dataloader.dataset.ConfusionMatrix}")
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'    
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)             
            if args.Commonalities:
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
    ML = True
    logger.info("***** Evaluation file name:{} *****".format(file_name))
    query_dataset = TextDataset(tokenizer, args, file_name, pool, ML=ML, NL_LT=args.NL_LT)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)

    if ML and not evalML:
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
        codebase_file = "dataset/ML/train.jsonl"
    code_dataset = TextDataset(tokenizer, args, codebase_file, pool, ML=ML, NL_LT=args.NL_LT)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    nl_urls=[]
    code_urls=[]
    # nl_lengths = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)  


    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        if args.Commonalities:
            CRModel = torch.nn.DataParallel(CRModel)
        else:
            model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.Commonalities:
        CRModel.eval()
        # model = CRModel.model
    else:
        model.eval()
    code_vecs, code_embs, lt_embs=[], [], [] 
    nl_vecs=[]
    logger.info("start query encoding...")
    for batch in query_dataloader:  
        if "graphcodebert" in args.model_name_or_path:
            nl_inputs = batch[3].to(args.device)
        else:
            nl_inputs = batch[2].to(args.device)
        with torch.no_grad():
            with autocast():
                if args.Commonalities:
                    nl_vec = CRModel(nl_inputs=nl_inputs, mode='nl')[1]
                else:
                    nl_vec = model(nl_inputs=nl_inputs)[1]
                nl_vecs.append(nl_vec.detach().cpu()) 
    batch_idx = 0
    logger.info("start code encoding...")
    if evalML and os.path.exists(f"{args.model_type}_code_vecs_r{rounds}.pkl"):
        code_vecs = pickle.load(open(f"{args.model_type}_code_vecs_r{rounds}.pkl", "rb"))
        code_lang_labels = pickle.load(open(f"code_lang_labels.pkl", "rb"))
    else:
        code_lang_labels, lang_sims = [], []
        for batch in code_dataloader:
            if "graphcodebert" in args.model_name_or_path:
                code_inputs = batch[0].to(args.device)    
                attn_mask = batch[1].to(args.device)
                position_idx =batch[2].to(args.device)
                lang_labels = batch[4].to(args.device)
            else:
                code_inputs = batch[0].to(args.device)  
                position_idx = batch[1].to(args.device)
                lang_labels = batch[3].to(args.device)
                attn_mask = None
            code_lang_labels.extend(lang_labels.cpu().tolist())
            with torch.no_grad():
                with autocast():
                    CLS_emb, LT_emb = None, None
                    if args.Commonalities:
                        results = CRModel(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx, nl_inputs=None, lang_labels=lang_labels, nl_length=None, mode="test")  
                        lang_sim = results["lang_sim"]
                        lang_sims.append(lang_sim.detach().cpu())
                        if results["fusion_emb"] is not None:
                            code_vec = results["fusion_emb"]
                        else:
                            code_vec = results["LI_emb"]
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
        
        if args.Commonalities:
            CRModel.train()
        else:
            model.train() 
        if code_vecs != []:
            code_vecs=torch.cat(code_vecs,0).float()
        else:
            code_embs=torch.cat(code_embs,0).float()
            lt_embs=torch.cat(lt_embs,0).float()
        if evalML:
            with open(f"{args.model_type}_code_vecs_r{rounds}.pkl", "wb") as f:
                pickle.dump(code_vecs, f)
            with open(f"code_lang_labels.pkl", "wb") as f:
                pickle.dump(code_lang_labels, f)
    nl_vecs=torch.cat(nl_vecs,0).float()
    lang_sims = torch.cat(lang_sims, 0).float()
    logger.info("lang simiarity: {}".format(lang_sims.mean(dim=0).tolist()))
    logger.info("start scores computing...")
    NBS = 256
    CodeLength = len(code_dataset)
    scores = []
    if code_vecs != []:
        scores=np.matmul(nl_vecs,code_vecs.T)  
    else:
        scores=np.maximum(np.matmul(nl_vecs,code_embs.T), np.matmul(nl_vecs,lt_embs.T))  
    
    topN = 10
    scores = scores.cpu().numpy()
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    # each nl to sort from big to small scores

    ranks=[]
    lang_distr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for i, idx in enumerate(sort_id[:1000]):
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
            if i < topN:
                lang_distr[code_lang_labels[idx]] += 1/rank
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks)),
        "lang_distr": lang_distr
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
    parser.add_argument("--tokenizer_type", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--summaryOper", default="NAR", type=str,
                        help="generate summary method")
    
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
    parser.add_argument('--MultiLT', action='store_true',
                help="Use multi language tokens.")
    parser.add_argument('--NL_LT', action='store_true',
                help="NL Use language tokens.")
    parser.add_argument('--BSampler', action='store_true',
                help="Use Batch sampler.")
    parser.add_argument('--Commonalities', action='store_true',
                help="Use code Commonalities as extra task.")
    parser.add_argument('--ACTION', action='store_true',
                help="")
    parser.add_argument('--CN', action='store_true',
                help="")
    parser.add_argument('--MCL', action='store_true',
                help="Use Contrastive learning.")
    parser.add_argument('--evalML', action='store_true',
                help="testing in ML codebase.")

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
    pool = multiprocessing.Pool(cpu_cont)

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    args.n_gpu = 1
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    if "graphcodebert" in args.model_name_or_path:
        args.model_type = "graphcodebert"
    elif "codebert" in args.model_name_or_path:
        args.model_type = "codebert"
    elif "unixcoder" in args.model_name_or_path:
        args.model_type = "unixcoder"

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 

    tokenizer.add_tokens(LANGS_TOKENS)
    model.resize_token_embeddings(len(tokenizer))
    CLS_embedding = model.embeddings.word_embeddings.weight[tokenizer.cls_token_id]
    for lang_token in LANGS_TOKENS:
        new_token_id = tokenizer.convert_tokens_to_ids(lang_token)
        model.embeddings.word_embeddings.weight.data[new_token_id] = CLS_embedding
    model=Model(model, args)
    DCmodel = None
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    if args.Commonalities:
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
        if args.Commonalities:
            CRModel.to(args.device)
            CRModel.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)
        model.to(args.device)
        model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
        
        result=evaluate(args, model, tokenizer, args.eval_data_file, pool, CRModel=CRModel, evalML=args.evalML)
         
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if args.Commonalities:
            CRModel.to(args.device)
            CRModel.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
            result=evaluate(args, model, tokenizer, args.test_data_file, pool, CRModel=CRModel, evalML=args.evalML)
        else:
            model.to(args.device)
            model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
            
            result=evaluate(args, model, tokenizer,args.test_data_file, pool, evalML=args.evalML)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()
