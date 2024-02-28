### Building Blocks of Transformers with PyTorch
This repository implements the core building blocks of Transformers, a powerful neural network architecture for various natural language processing (NLP) tasks, using the PyTorch deep learning framework.

What's Inside?
This repository provides implementations of the following key Transformer components:

---> Transformer Encoder: Processes and encodes input sequences.

---> Transformer Decoder: Decodes encoded information to generate output sequences.

---> Multi-Head Attention: Extends the attention mechanism with multiple parallel attention heads for richer representation learning.

---> Absolute Positional Encoding: Injects positional information into the model since Transformers lack inherent knowledge of sequence order.


Getting Started
### Prerequisites:

---> Python 3

---> PyTorch: Install using pip install torch


### Notations used
| Symbol        | Meaning       | 
| ------------- |:-------------:| 
| num_queries or max_sequence_length | maximum length of the sequence| 
| head_dims or d_k or d_v | diension of key vector and value vector|   
| num_heads or h | number of self attention working independantly|  
