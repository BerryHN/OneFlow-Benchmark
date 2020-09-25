#Debug Edition:
# debug = True only works with eager mode
# Author： Jiachen

## TODO: FInish Train
## TODO: Preidiction
## TODO: Add gradient clip
## TODO: Add optimizer decay

import torch_mingpt

import oneflow as flow
import oneflow.typing as tp
import math
import numpy as np


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

def Causal_Self_Attention(x, config,name='csa'):
    """
    Input:: 
        x : Eembedded words input[B, T, C]
            -- B is the batch size
            -- T is the sequence length(block_size)
            -- C is the dimension of the embedding (n_embd)
               C/head_number = dimension of each head(d_k)
        config: class object defined with models.GPTConfig
    Output::
        y : output of x, which can be used as new x in next interation
    
 
    Description::
        This functions is the causl_sefl_attention core, which is a part of multiple head attention
        schema.
        Code refered from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        Theory refered from: http://jalammar.github.io/illustrated-gpt2/
        Related paper: 
    """
    assert config.n_embd % config.n_head == 0

    #def 
    B,T,C = x.shape
    #Kaiming_initialize 
    kaiming_init_C = flow.kaiming_initializer(shape=(C, C))
    ## calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # define: key, query and value projections for all heads 
    # process: query + key ----> value 
    # dimension: (B,T,C) -> (B, nh, T, hs), nh*ns=C
    
    # query:The query is a representation of the current word used to score against all the other words (using their keys). 
    query = flow.layers.dense(x,units=config.n_embd, 
                    kernel_initializer=kaiming_init_C, name=(name+'_query'))
    query = flow.reshape(query,[B,T, config.n_head, C//config.n_head])
    query = flow.transpose(query,[0,2,1,3])
    # key:Key vectors are like labels for all the words in the segment. 
    key = flow.layers.dense(x,units=config.n_embd, kernel_initializer=kaiming_init_C, name=(name+'_key'))
    key = flow.reshape(key,[B,T, config.n_head, C//config.n_head])
    key = flow.transpose(key,[0,2,1,3])
    # value: Value vectors are actual word representations
    value = flow.layers.dense(x,units=config.n_embd, kernel_initializer=kaiming_init_C, name=(name+'value'))
    value = flow.reshape(value,[B,T, config.n_head, C//config.n_head])
    value = flow.transpose(value,[0,2,1,3])
    

    ##causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = flow.matmul(query, flow.transpose(key,[0,1,3,2])) *(1.0 / math.sqrt(key.shape[-1]))
    att_tril = flow.math.tril(flow.constant(value=int(-1), 
                        dtype=flow.int32, shape=(B,config.n_head,T,T),
                        name=name + "_ConstantLike_tril"))
    att_tril = att_tril+ flow.ones_like(like=att_tril,dtype=flow.int32)
    att = flow.masked_fill(att, att_tril, float('-inf'))
    att = flow.nn.softmax(att, name= name+'att')
    att = flow.nn.dropout(att,config.attn_pdrop)
    ## QK*V: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = flow.matmul(att,value)
    y = flow.transpose(y,[0,2,1,3])
    y = flow.reshape(y,[B,T,C])
    y = flow.nn.dropout(y, config.resid_pdrop)
    return y


def Block(x, config, name='Block_'):
    #kaiming_init_C = flow.kaiming_initializer(shape=(C, C))
    #attn of X
    x = flow.layers.layer_norm(x, name=name+'l1')
    x = x + Causal_Self_Attention(x, config, name = name+'attentions')
    #mlp
    x = flow.layers.layer_norm(x, name=name+'l2')
    x = flow.layers.dense(inputs=x, units=4*config.n_embd,
                         kernel_initializer=flow.kaiming_initializer(shape=(config.n_embd,4*config.n_embd)),
                         activation=flow.math.gelu, name = name+'gelu')
    x = flow.layers.dense(inputs=x, units=config.n_embd,
                         kernel_initializer=flow.kaiming_initializer(shape=(4*config.n_embd,config.n_embd)), name = name+'dense'
                         )
    x = flow.nn.dropout(x,rate=config.resid_pdrop)

    return x

def GPT(idx,config, target=None):
    b,t = idx.shape
    assert t <= config.block_size, "Cannot forward, model block size is exhausted."

    
    #forward the GPT model
    #token_embeddings = flow.layers.dense
    word_embedding = flow.get_variable('word_emb', initializer=flow.random_normal_initializer(),
                                shape=(config.vocab_size,config.n_embd))
    token_embeddings = flow.gather(word_embedding, idx)
    
    #positions embedding
    pos_emb = flow.get_variable(name='pos_emb',
                         shape=(1,config.block_size,config.n_embd),dtype=flow.float32,
                         initializer = flow.zeros_initializer())
    #position_embeddings = fpos_emb[:, :t, :] # each position maps to a (learnable) vector
    position_embeddings = flow.slice(pos_emb,[None,0,None],[None,t,None])
    x = flow.nn.dropout((token_embeddings+position_embeddings), config.embd_pdrop)
    
    #Blocks
    for block_id in range(config.n_layer):
        with flow.scope.namespace('Block'+str(block_id)):
            x = Block(x,config)
            
    x = flow.layers.layer_norm(x,name='output_layernorm')

    logits = flow.layers.dense(x, config.vocab_size, use_bias=False,
                                activation= flow.zeros_initializer(),
                                name = 'output_logits')
    
    loss = None
    if target is not None:
        #TODO 
        logits = flow.reshape(logits, [-1,config.vocab_size])
        target = flow.reshape(target,[-1])
        target = flow.one_hot(target,depth=config.vocab_size,dtype=flow.float32)
        loss = flow.nn.softmax_cross_entropy_with_logits(logits,target)
    return logits,loss