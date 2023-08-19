import copy
import math
import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
from torch.autograd import Function
# from transformers import MultiHeadAttention
LANGS = ['go', 'java', 'javascript', 'php', 'python', 'ruby']

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

    def get(self,x):
        return self.pe(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # with torch.no_grad():
        #     x = (p_attn[0,0,-1,:]*1000).long()
        #     x = (p_attn[0,0,-2,:]*1000).long()
        #     print(x.tolist())
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MyMultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MyMultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """

    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MyMultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter, dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.encoder = nn.ModuleList([TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for _ in range(6)])

    def forward(self, x, mask):
        for i, block in enumerate(self.encoder):
            x = block(x, mask)
        # x = self.encoder(x, mask)
        return x
    
def spectralNorm(W, n_iteration=5):
    """
        Spectral normalization for Lipschitz constrain in Disc of WGAN
        |W|^2 = principal eigenvalue of W^TW through power iteration
        v = W^Tu/|W^Tu|
        u = Wv / |Wv|
        |W|^2 = u^TWv
        
        :param w: Tensor of (out_dim, in_dim) or (out_dim), weight matrix of NN
        :param n_iteration: int, number of iterations for iterative calculation of spectral normalization:
        :returns: Tensor of (), spectral normalization of weight matrix
    """
    device = W.device
    # (o, i)
    # bias: (O) -> (o, 1)
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    out_dim, in_dim = W.size()
    # (i, o)
    Wt = W.transpose(0, 1)
    # (1, i)
    u = torch.ones(1, in_dim).to(device)
    for _ in range(n_iteration):
        # (1, i) * (i, o) -> (1, o)
        v = torch.mm(u, Wt)
        v = v / v.norm(p=2)
        # (1, o) * (o, i) -> (1, i)
        u = torch.mm(v, W)
        u = u / u.norm(p=2)
    # (1, i) * (i, o) * (o, 1) -> (1, 1)
    sn = torch.mm(torch.mm(u, Wt), v.transpose(0, 1)).sum() ** 0.5
    return sn


        
class Model(nn.Module):   
    def __init__(self, encoder, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.args = args
        self.output = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768))
      
    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None, past_key_values=None, prefix_attention_mask=None, CR=False): 
        if code_inputs is not None:
            if "graphcodebert" in self.args.model_name_or_path:
            # if False:
                nodes_mask=position_idx.eq(0)
                token_mask=position_idx.ge(2)        
                inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
                nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
                nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]  

                avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
                inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
                outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, past_key_values=past_key_values, output_hidden_states=True if CR else False)
            else:
                if prefix_attention_mask is not None:
                    attn_mask = code_inputs.ne(1)
                    attn_mask = torch.cat([prefix_attention_mask, attn_mask], dim=1)
                outputs = self.encoder(code_inputs,attention_mask=attn_mask, past_key_values=past_key_values, output_hidden_states=True if CR else False)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))
        return outputs
    


class NLAttention(nn.Module):
    def __init__(self, d_model, d_key, d_value, length=16, dropout=0.1):
        super(NLAttention, self).__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.length = length
        self.dropout = nn.Dropout(dropout)
        self.q_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(length)])

        self.out = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        q = q.unsqueeze(1).expand(-1, self.length, -1).clone()
        for i, q_linear in enumerate(self.q_linears):
            q[:,i,:] = q_linear(q[:,i,:])

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_key)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand_as(weights)
            weights = weights.masked_fill(mask == 0, torch.finfo(weights.dtype).min)

        weights = self.softmax(weights)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        output = self.out(output)
        return output


class FeatureExtraction(nn.Module):
    def __init__(self, d_model=768, d_key=768, d_value=768, length=8, dropout=0.1):
        super(FeatureExtraction, self).__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.length = length
        self.dropout = nn.Dropout(dropout)
        self.q_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(length)])

        self.out = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # k = self.k_linear(k)
        # expand q length times: (batch, d_model) -> (batch, length, d_model)
        if self.length == 1:
            q = self.q_linears[0](q)
            q = q.unsqueeze(1)
        else:
            q = q.unsqueeze(1).expand(-1, self.length, -1).clone()
            for i, q_linear in enumerate(self.q_linears):
                q[:,i,:] = q_linear(q[:,i,:])
        # q = q.view(-1, self.length, self.d_model)  # (batch, length * d_model) -> (batch, length, d_model)
        # v = self.v_linear(v)
        # 给定多个个序列向量，每个序列长度不同，如何将这些长度不同的向量的长度统一（如有4个[32,768]的向量，其真实长度为[15, 26, 18, 10], 如何将其统一成长度16的向量,即4个[16,768]向量）

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_key)  # (batch, d_key) * (batch, length, d_key) -> (batch, length, length)
        # print(f"q: {q.shape} k: {k.shape} weights: {weights.shape}")

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand_as(weights)
            # print(f"weights: {weights.shape} mask: {mask.shape}")
            weights = weights.masked_fill(mask == 0, torch.finfo(weights.dtype).min)

        weights = self.softmax(weights)
        # print(f"weights: {weights.shape}")
        weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        output = self.out(output)
        # print(f"output: {output.shape}")
        if self.length == 1:
            output = output.squeeze(1)
        return output

class MLTModule(nn.Module):
    def __init__(self, lang):
        super(MLTModule, self).__init__()
        self.lang = lang 
        self.langIdx = LANGS.index(lang)
        self.n_dim = 768
        # self.lstm = nn.LSTM(768, 768, batch_first=True)
        # self.linear = nn.Linear(self.n_dim, self.n_dim)
        self.out = nn.Linear(self.n_dim * 2, self.n_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        idx = self.langIdx
        langEmb = inputs[:,idx,:].unsqueeze(1)
        # langEmb = self.linear(langEmb)
        attention = torch.matmul(langEmb, inputs.transpose(1, 2))  / math.sqrt(self.n_dim)
        attention[:, :, idx] = torch.finfo(attention.dtype).min
        attention_weights = F.softmax(attention, dim=-1)
        attended_vectors = torch.matmul(attention_weights, inputs)
        fusionEmb = self.out(torch.cat([langEmb, attended_vectors], dim=-1))
        return fusionEmb.squeeze(1)
    
class MLTModuleV2(nn.Module):
    def __init__(self):
        super(MLTModuleV2, self).__init__()
        self.n_dim = 768
        # self.lstm = nn.LSTM(768, 768, batch_first=True)
        self.linear = nn.Linear(self.n_dim, self.n_dim)
        self.out = nn.Linear(self.n_dim * 2, self.n_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, idx):
        originLangEmb = inputs[:,idx,:].unsqueeze(1)
        langEmb = self.linear(originLangEmb)
        attention = torch.matmul(langEmb, inputs.transpose(1, 2))  / math.sqrt(self.n_dim)
        attention[:, :, idx] = torch.finfo(attention.dtype).min
        attention_weights = F.softmax(attention, dim=-1)
        attended_vectors = torch.matmul(attention_weights, inputs)
        fusionEmb = self.out(torch.cat([originLangEmb, attended_vectors], dim=-1))
        return attention_weights, fusionEmb.squeeze(1)
    
    
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, prefix_projection=True, num_hidden_layers=12, hidden_size=768, prefix_hidden_size=768, pre_seq_len=6):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class CRNet(nn.Module):
    def __init__(self, args, model, CSModel, PCModel, summaryOper='NAR') -> None:
        super(CRNet, self).__init__()
        
        self.args = args
        self.model = model

        self.pooler = nn.Sequential(self.model.encoder.pooler.dense,
                                    self.model.encoder.pooler.activation,)

        self.CSLoss = nn.MSELoss()  
        self.Fusion = nn.Sequential(nn.Linear(768*3, 768), 
                                    nn.Tanh())
        
        if args.MultiLT:
            self.MultiLT = MLTModuleV2()
        
    def get_adversarial_result(self, x, label, alpha=0.01):
        loss_fn = nn.CrossEntropyLoss()
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, label)
        return loss_adv
    
    def shuffle_in_batch(self, x):
        device = x.device
        batch_size, seq_len, _ = x.size()
        if batch_size == 1:
            sx = x
        else:
            batch_index = torch.arange(batch_size).unsqueeze(1)
            shuffle_index = (torch.arange(batch_size - 1) + 1).repeat(seq_len // (batch_size - 1) + 1)[:seq_len].unsqueeze(0)
            batch_index = batch_index.to(device)
            shuffle_index = shuffle_index.to(device)
            shuffled_batch_index  = (batch_index + shuffle_index) % batch_size
            seq_batch_index = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            seq_batch_index = seq_batch_index.to(device)
            sx = x[shuffled_batch_index, seq_batch_index]
        return sx
    
    def MI(self, lang_emb, cls_emb):
        # batch_size * seq_len * cls_emb_dim
        seq_len = 6
        cls_emb = cls_emb.unsqueeze(1).expand(-1, seq_len, -1)
        s_cls_emb = self.shuffle_in_batch(cls_emb)
        seq_mi = -F.softplus(-self.disc(lang_emb, cls_emb)).sum(dim=0) / seq_len - F.softplus(self.disc(lang_emb, s_cls_emb)).sum(dim=0) / seq_len
        mi = seq_mi.mean()
        return mi

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None, lang_labels=None, nl_length=None, step=0, mode="train"):
        if mode == 'nl':
            return self.model(nl_inputs=nl_inputs)
        query_emb, LI_emb, LS_emb, fusion_emb, adv_loss, cs_loss, length_loss = None, None, None, None, None, None, None
        BS = code_inputs.shape[0]
        code_emb = self.model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx, CR=False)
        code_length = torch.sum(code_inputs.ne(1), dim=1)

        CLS_emb = code_emb[1]

        if self.args.MultiLT:

            LT_emb = torch.zeros_like(code_emb.last_hidden_state[:, 1, :]).half()
            LANG_emb = torch.zeros_like(code_emb.last_hidden_state[:, 1, :]).half()
            lang_sim = torch.zeros((BS, 6)).half().to(self.args.device)
            for i in range(6):
                lang_emb = code_emb.last_hidden_state[lang_labels==i, 1:7, :]
                lang_emb = self.pooler(lang_emb)
                if lang_emb.shape[0] == 0:
                    continue
                _, pw_emb = self.MultiLT(lang_emb, i)
                if LT_emb is not None:
                    LT_emb[lang_labels==i] = pw_emb[:,i,:].squeeze(1)
                LANG_emb[lang_labels==i] = lang_emb[:,i,:].squeeze(1)
        else:
            pass

        cs_loss = self.MI(lang_emb, CLS_emb)
        if self.args.MultiLT:
            LI_emb = LT_emb
        else:
            LI_emb = CLS_emb
        if mode == 'train':
            nl_outputs = self.model(nl_inputs=nl_inputs)
            query_emb = nl_outputs[1]

        fusion_emb = self.Fusion(torch.cat([CLS_emb, LI_emb, LANG_emb], dim=1))
        results = {"query_emb": query_emb, "LI_emb": LI_emb, "LS_emb": LS_emb, "fusion_emb":fusion_emb, "adv_loss": adv_loss, "cs_loss": cs_loss, 'length_loss':length_loss, 'lang_sim': lang_sim}
        return results

