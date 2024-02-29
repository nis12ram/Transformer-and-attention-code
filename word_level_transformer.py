import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as f

  
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class AbsolutePositionalEncoding(nn.Module):
  def __init__(self,d_model,max_sequence_length):
    super().__init__()
    self.d_model = d_model
    self.max_sequence_length = max_sequence_length
  def forward(self):
    even_i = torch.arange(0,self.d_model,2)  # shape (d_model//2)
    denominator = torch.pow(10000,even_i/self.d_model).reshape(1,self.d_model//2) # shape (1,d_model//2)
    denominator = torch.reciprocal(denominator) # shape (1,d_model//2)
    position = torch.arange(self.max_sequence_length,dtype = torch.float).reshape(self.max_sequence_length,1) # shape (num_queries,1)
    angle_values = torch.matmul(position,denominator) # shape (num_queries,d_model//2)
    even_PE = torch.sin(angle_values) # shape (num_queries,d_model//2)
    odd_PE = torch.cos(angle_values)  # shape (num_queries,d_model//2)
    stacked = torch.stack([even_PE,odd_PE],dim = -1)  # shape (num_queries,d_model//2,2)
    PE = torch.reshape(stacked,(self.max_sequence_length,self.d_model)) # shape (max_sequence_length,d_model)
    return PE

class SentenceEmbedding(nn.Module):
  def __init__(self,max_sequence_length,d_model,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
    super().__init__()
    self.vocab_size = len(language_to_index)
    self.max_sequence_length = max_sequence_length
    self.embedding = nn.Embedding(self.vocab_size,d_model)  # (len_vocan,embedded_dim) {embedded_dim = d_model}
    self.language_to_index = language_to_index
    self.position_encoder = AbsolutePositionalEncoding(d_model=d_model,max_sequence_length=max_sequence_length)
    self.dropout = nn.Dropout(p = 0.1)
    self.START_TOKEN = START_TOKEN
    self.END_TOKEN = END_TOKEN
    self.PADDING_TOKEN = PADDING_TOKEN


  def sentence_tokenize(self,sentence,start_token,end_token):
      '''
      sentence is preprocessed(using text_preprocessing()) so its each token would be their in language_to_index dictionary.
      '''
      # print(sentence)
      sentence_word_indices = []
      for token in sentence.split():
        '''
        if token in vocabulary then append the index 
            and if not in vocabulary then ignore the word
        '''
        try:
            sentence_word_indices.append(self.language_to_index[token])
        except:
          pass
            

      if start_token:
        sentence_word_indices.insert(0,self.language_to_index[self.START_TOKEN])
      if end_token:
        sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
      
      for _ in range(len(sentence_word_indices),self.max_sequence_length):
        sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
      # now len(sentence_word_indices) = max_sequence_length
      # print(sentence_word_indices)
      return torch.tensor(sentence_word_indices)
  
 

  def batch_tokenize(self,batch,start_token ,end_token):
    '''
    Here batch:
              it is a tuple ,
              len(batch) = batch_size ,
              contain sentences of vocabulary language_to_index
              example: if language_to_index is of english and batch_size = 3
                      then batch = ("i am happy","let's go","let's play")
    '''

    tokenized = []
    for sentence_num in range(len(batch)):
      tokenized.append(self.sentence_tokenize(batch[sentence_num],start_token,end_token))
    tokenized = torch.stack(tokenized)  # shape (batch_size,max_sequence_length) {max_sequence_length means same as num_queries}
    return tokenized

  def forward(self,x,start_token,end_token):
    '''
    Here x:
           tuple of sentences of vocabulary language_to_index
    '''
    ### step 1 convert batch sentence to batch index(tensor)
    x = self.batch_tokenize(x,start_token,end_token) # shape (batch_size,num_queries)

    x = self.embedding(x) # shape (batch_size,num_queries,d_model)
    pe = self.position_encoder.forward() # shape (num_queries,d_model)
    x = x + pe # shape (batch_size,num_queries,d_model)
    x = self.dropout(x) # shape (batch_size,num_queries,d_model)
    return x


class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()

    self.sequence_length = None
    self.batch_size = None
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dims = self.d_model // self.num_heads # head_dims = d_k(dimension of key vector) = d_v(dimension of value vector)
    self.qkv_layer = nn.Linear(in_features = self.d_model,out_features = 3*self.d_model)
    self.linear_layer = nn.Linear(in_features = self.d_model,out_features = self.d_model)

  def scaled_dot_product_attention(self,q,k,v,mask = None):

    '''
    q shape (batch_size,num_heads,num_queries,head_dims)
    k shape (batch_size,num_heads,num_kv,head_dims)
    v shape (batch_size,num_heads,num_kv,head_dims)

    num_kv - number of key value pair whome you want to use to compute attentional representation
    num_queries - max sequence length
    Here: (num_kv = num_queries) {attention is paid to whole sequence to compute attention representation of a specific input in a sequence}
    num_heads - number of attention heads
    head_dims - dimension of key vector(d_k) and value vector(d_v) {d_k = d_v}
    '''
    d_k = self.head_dims
    scaled = torch.matmul(q,k.transpose(-2,-1)) / np.sqrt(d_k)  # shape (batch_size,num_heads,num_queries,num_kv) (num_queries == num_kv)
    if (mask is not None):
      # print(f'scaled {scaled.shape}')
      
      mask = mask.reshape(mask.size()[0],1,mask.size()[1],mask.size()[2]) # shape (batch_size,1,num_queries,num_queries)
      # print(f'mask {mask.shape}')
      scaled += mask
    attention = f.softmax(scaled,dim = -1) # shape (batch_size,num_heads,num_queries,num_kv) (num_queries == num_kv)
    values = torch.matmul(attention,v) # shape (batch_size,num_heads,num_queries,head_dims) (head_dims = d_v)
    return values,attention



  def forward(self,x,mask = None):
    self.batch_size = x.size()[0]
    self.sequence_length = x.size()[1]
    qkv = self.qkv_layer(x) # shape (batch_size,num_queries,3*d_model)
    qkv = qkv.reshape(self.batch_size,self.sequence_length,self.num_heads,3*self.head_dims) # shape (batch_size,num_queries,num_heads,3*head_dims)
    qkv = torch.permute(qkv,(0,2,1,3))  # shape (batch_size,num_heads,num_queries,3*head_dims)
    q,k,v = torch.chunk(qkv,3,dim= -1)  # each shape (batch_size,num_heads,num_queries,head_dims)
    values,attention = self.scaled_dot_product_attention(q,k,v,mask = mask)
    values = values.reshape(self.batch_size,self.sequence_length,self.head_dims * self.num_heads) # shape (batch_size,num_queries,head_dims * num_heads)
    out = self.linear_layer(values) # shape (batch_size,num_queries,d_model)
    return out



class LayerNormalization(nn.Module):
  def __init__(self,parameter_shape,eps = 1e-5):
    '''
    parameter_shape - represents along which dimension you want to normalize
    eps - epsilon for numerical stability
    gamma and beta are learnable parameters

    example: input ---> (batch_size,num_queries,d_model)
            and if you want to normalize along last dimension
            ---> then parameter_shape = (d_model,)
            ---> then gamma and beta shape = (d_model,)

    '''


    super().__init__()
    self.parameter_shape = parameter_shape
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(self.parameter_shape))
    self.beta = nn.Parameter(torch.zeros(self.parameter_shape))
  def forward(self,inputs):

    '''
    inputs shape (batch_size,num_queries,d_model)
    equation to code is ---> gamma * (inputs-mean)/std * beta
    '''
    dims = [-(i+1) for i in range(len(self.parameter_shape))] # len(parameter_shape) = 1 then dims = [-1]
    mean = torch.mean(inputs,dim = dims,keepdim = True) # shape (batch_size,num_queries,1)
    '''
    inputs-mean shape is (batch_size,num_queries,d_model) { column-wise boroadcasting occurs }
    '''

    var = ((inputs-mean)**2).mean(dim = dims,keepdim = True)  # shape (batch_size,num_queries,1)
    # print(f'what we need is {var.shape}')
    std = torch.sqrt(var+self.eps)  # shape (batch_size,num_queries,1)
    y = (inputs-mean)/std # shape (batch_size,num_queries,d_model) {due to broadcasting in column}
    out = self.gamma * y + self.beta  # shape (batch_size,num_queries,d_model) {here also boroadcasting happens}
    return out

class PositionalwiseFeedForwrd(nn.Module):

  def __init__(self,d_model,hidden,drop_prob):
    super().__init__()
    self.linear1 = nn.Linear(in_features = d_model,out_features = hidden)
    self.linear2 = nn.Linear(in_features = hidden,out_features = d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p = drop_prob)

  def forward(self,x):
    '''
    eqaution to code is ---> linear2(Dropout(relu(linear1(x))))
    '''
    x = self.linear1(x) # shape (batch_size,num_queries,hidden)
    x = self.relu(x)  # shape (batch_size,num_queries,hidden)
    x = self.dropout(x) # shape (batch_size,num_queries,hidden)
    x = self.linear2(x) # shape (batch_size,num_queries,d_model)
    return x


class EncoderLayer(nn.Module):

  def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):

    super().__init__()
    self.attention = MultiHeadAttention(d_model = d_model,num_heads = num_heads)
    self.norm1 = LayerNormalization(parameter_shape=(d_model,))
    self.dropout = nn.Dropout(p = drop_prob)
    self.ffn = PositionalwiseFeedForwrd(d_model = d_model,hidden = ffn_hidden,drop_prob = drop_prob)
    self.norm2 = LayerNormalization(parameter_shape = (d_model,))

  def forward(self,x,self_attention_mask):
    '''
    x shape (batch_size,num_queries,d_model)
    '''
    residual_x = x  # shape (batch_size,num_queries,d_model)
    x = self.attention.forward(x,mask = self_attention_mask) # shape (batch_size,num_queries,d_model)
    x = self.dropout(x) # shape (batch_size,num_queries,d_model)
    x = self.norm1.forward(x + residual_x) # shape (batch_size,num_queries,d_model)
    residual_x = x  # shape (batch_size,num_queries,d_model)
    x = self.ffn.forward(x) # shape (batch_size,num_queries,d_model)
    x = self.dropout(x) # shape (batch_size,num_queries,d_model)
    x = self.norm2.forward(x + residual_x)  # shape (batch_size,num_queries,d_model)
    return x


class SequentialEncoder(nn.Sequential):
    '''
    overriding the forward method because by default forward method consider one input parameter for the module but here we need to put two input parameter.
    '''
    def forward(self,*inputs):
        
        x,self_attention_mask = inputs
        for module in self._modules.values():
            x = module.forward(x,self_attention_mask)
            
        return x


class Encoder(nn.Module):
  def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_stacked,max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN ):

    super().__init__()
    self.sentence_embedding = SentenceEmbedding(max_sequence_length,d_model,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN)

    # ***Important {sequential stacking of all encoder layers using nn.Sequential }
    self.layers = SequentialEncoder(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob)
                                  for _ in range(num_stacked)])
  def forward(self,x,self_attention_mask, start_token, end_token):
    '''
    x shape (batch_size,num_queries,d_model) {input embeddings + Positional Embeddings}
    '''
    
    x = self.sentence_embedding(x,start_token,end_token) # shape (batch_size,num_queries,d_model)
    x = self.layers(x,self_attention_mask) # shape (batch_size,num_queries,d_model)
    return x



class MultiHeadCrossAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()

    self.sequence_length = None
    self.batch_size = None
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dims = self.d_model // self.num_heads # head_dims = d_k(dimension of key vector) = d_v(dimension of value vector)
    self.kv_layer = nn.Linear(in_features = self.d_model,out_features = 2 * self.d_model)
    self.q_layer = nn.Linear(in_features = self.d_model,out_features = self.d_model)
    self.linear_layer = nn.Linear(in_features = self.d_model,out_features = self.d_model)

  def scaled_dot_product_attention(self,q,k,v,mask = None):

    '''
    q shape (batch_size,num_heads,num_queries,head_dims)
    k shape (batch_size,num_heads,num_kv,head_dims)
    v shape (batch_size,num_heads,num_kv,head_dims)

    num_kv - number of key value pair whome you want to use to compute attentional representation
    num_queries - max sequence length
    Here: (num_kv = num_queries) {attention is paid to whole sequence to compute attention representation of a specific input in a sequence}
    num_heads - number of attention heads
    head_dims - dimension of key vector(d_k) and value vector(d_v) {d_k = d_v}
    '''
    d_k = self.head_dims
    scaled = torch.matmul(q,k.transpose(-2,-1)) / np.sqrt(d_k)  # shape (batch_size,num_heads,num_queries,num_kv) (num_queries == num_kv)
    if (mask is not None):
      mask = mask.reshape(mask.size()[0],1,mask.size()[1],mask.size()[2]) # shape (batch_size,1,num_queries,num_queries
      scaled += mask  # shape (batch_size,num_heads,num_queries,num_kv) (num_queries == num_kv)
    attention = f.softmax(scaled,dim = -1) # shape (batch_size,num_heads,num_queries,num_kv) (num_queries == num_kv)
    values = torch.matmul(attention,v) # shape (batch_size,num_heads,num_queries,head_dims) (head_dims = d_v)
    return values,attention


  def forward(self,x,y,mask = None):
    '''
    x shape (batch_size,num_queries,d_model) {x represent output of top most encoder stacked layer}
    y shape (batch_size,num_queries,d_model) {y represent output of add and norm block 1 of decoder layer}
    '''
    self.batch_size = x.size()[0]
    self.sequence_length = x.size()[1]
    q = self.q_layer(y) # shape (batch_size,num_queries,d_model)
    kv = self.kv_layer(x) # shape (batch_size,num_queries,2*d_model)
    q = q.reshape(self.batch_size,self.sequence_length,self.num_heads,self.head_dims) # shape (batch_size,num_queries,num_heads,head_dims)
    kv = kv.reshape(self.batch_size,self.sequence_length,self.num_heads,2 * self.head_dims) # shape (batch_size,num_queries,num_heads,2 * head_dims)
    q = torch.permute(q,(0,2,1,3))  # shape (batch_size,num_heads,num_qureies,head_dims)
    kv = torch.permute(kv,(0,2,1,3))  # shape (batch_size,num_heads,num_queries,2 * head_dims)
    k, v = torch.chunk(kv,2,dim = -1) # each shape (batch_size,num_heads,num_queries,head_dims)
    values,attention = self.scaled_dot_product_attention(q,k,v,mask = mask)
    values = values.reshape(self.batch_size,self.sequence_length,self.d_model) # shape (batch_size,num_queries,num_heads *  head_dims) {concatentaing all heads together}
    out = self.linear_layer(values) # shape (batch_size,num_queries,d_model)
    return out

class DecoderLayer(nn.Module):
  def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
    super().__init__()
    self.masked_attention = MultiHeadAttention(d_model = d_model,num_heads = num_heads)
    self.dropout = nn.Dropout(p = drop_prob)
    self.norm1 = LayerNormalization(parameter_shape=(d_model,))
    self.encoder_decoder_attention = MultiHeadCrossAttention(d_model = d_model,num_heads = num_heads)
    self.norm2 = LayerNormalization(parameter_shape=(d_model,))
    self.ffn = PositionalwiseFeedForwrd(d_model = d_model,hidden = ffn_hidden,drop_prob = drop_prob)
    self.norm3 = LayerNormalization(parameter_shape=(d_model,))

  def forward(self,x,y,self_attention_mask,cross_attention_mask):
    '''
    x shape (batch_size,num_queries,d_model) {x represent output of top most encoder stacked layer}
    y shape (batch_size,num_queries,d_model) {y represent output of add and norm block 1 of decoder layer}
    '''
    residual_y = y  # shape (batch_size,num_queries,d_model)
    y = self.masked_attention.forward(y,mask = self_attention_mask)  # shape (batch_size,num_queries,d_model)
    y = self.dropout(y) # shape (batch_size,num_queries,d_model)
    y = self.norm1.forward(y + residual_y) # shape (batch_size,num_queries,d_model)

    residual_y = y  # shape (batch_size,num_queries,d_model)
    y = self.encoder_decoder_attention.forward(x,y,mask = cross_attention_mask) # shape (batch_size,num_queries,d_model)
    y = self.dropout(y) # shape (batch_size,num_queries,d_model)
    y = self.norm2.forward(y + residual_y)  # shape (batch_size,num_queries,d_model)

    residual_y = y  # shape (batch_size,num_queries,d_model)
    y = self.ffn.forward(y) # shape (batch_size,num_queries,d_model)
    y = self.dropout(y) # shape (batch_size,num_queries,d_model)
    y = self.norm2.forward(y + residual_y)  # shape (batch_size,num_queries,d_model)
    return y

class SequentialDecoder(nn.Sequential):
    '''
    SequentialDecoder class extends from nn.Sequential class:
                  whose __init__() needs the list of all decoder layers that are gonna stack on each other in order
                  example input to init: [DecoderLayer(...),DecoderLayer(...),DecoderLayer(...),...]


    _modules is a dictionary:
                  whose key represent the index of the decoder layer (type:str) in stack
                  whose value represent decoder layer(or Decoder Module)

                  example:stacked_layer = 3 then
                                  _modules = {'0':DecodeLayer(...),
                                              '1':DecoderLayer(...),
                                              '2':DecoderLayer(...)}

    module from these line(for module in self._modules...):
                  represent a specific decoder layer(or Decoder Module) from all off the stacked decoder layers(or Decoder Modules)



    forward method has been overrided:
                  because by default forward takes one argument but the transformer decoder takes three argument (x,y,mask).



    '''
    def forward(self, *inputs):
        x, y, self_attention_mask,cross_attention_mask = inputs
        # print(f'see the dict {self._modules.keys()}')
        for module in self._modules.values():
            y = module.forward(x, y, self_attention_mask,cross_attention_mask)  # shape (batch_size,num_queries,d_model)
        return y
    
    
class Decoder(nn.Module):
  def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_stacked,
               max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
    super().__init__()
    self.sentence_embedding = SentenceEmbedding(max_sequence_length,d_model,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN)
    self.layers = SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob)
                                        for  _ in range(num_stacked)])
  def forward(self,x,y,self_attention_mask, cross_attention_mask, start_token, end_token):

    '''
    x shape (batch_size,num_queries,d_model) {x represent output of top most encoder stacked layer}
    y shape (batch_size,num_queries,d_model) {y represent output of add and norm block 1 of decoder layer}
    mask shape (num_queries,num_queries) or (max_sequence_length,max_sequence_length)
    '''
    y = self.sentence_embedding(y,start_token,end_token)
    y = self.layers.forward(x,y,self_attention_mask,cross_attention_mask) # shape (batch_size,num_queries,d_model)
    return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_stacked,
                max_sequence_length, 
                sp_vocab_size,
                english_to_index,
                spanish_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        
        super().__init__()
        self.encoder = Encoder(d_model,ffn_hidden,num_heads,drop_prob,num_stacked,max_sequence_length,
                 english_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN )
        self.decoder = Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_stacked,max_sequence_length,
                 spanish_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN )
        
        self.linear = nn.Linear(d_model,sp_vocab_size)
        
    def forward(self,
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False):
        
        '''
        x :
           top most stacked encoder output
        y :
           batch of target sentence (type:tuple(str))
        '''
        
        x = self.encoder(x,encoder_self_attention_mask,start_token = enc_start_token,end_token = enc_end_token) # shape (batch_size,num_queries,d_moodel)
        out = self.decoder(x,y,decoder_self_attention_mask,decoder_cross_attention_mask,start_token = dec_start_token,end_token = dec_end_token) # shape (batch_size,num_queries,d_model)
        out = self.linear(out) # shape (batch_size,num_queries,targ_lng_vocab_size)
        return out
        
        
    






