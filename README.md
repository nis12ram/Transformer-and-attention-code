### Building Blocks of Transformers with PyTorch
This repository implements the core building blocks of Transformers, a powerful neural network architecture for various natural language processing (NLP) tasks, using the PyTorch deep learning framework.

What's Inside?
This repository provides implementations of the following key Transformer components:

---> Transformer Encoder: Processes and encodes input sequences.

---> Transformer Decoder: Decodes encoded information to generate output sequences.

---> Multi-Head Attention: Extends the attention mechanism with multiple parallel attention heads for richer representation learning.

---> Absolute Positional Encoding: Injects positional information into the model since Transformers lack inherent knowledge of sequence order.

---> translator model: uses transformer to build character level and word level language model.


Getting Started
### Prerequisites:

---> Python 3

---> PyTorch: Install using pip install torch


### Some Importnat Notations used
| Symbol        | Meaning       | 
| ------------- |:-------------:| 
| num_queries or max_sequence_length | maximum length of the sequence| 
| head_dims or d_k or d_v | diension of key vector and value vector|   
| num_heads or h | number of self attention working independantly|  


### Dataset used for spanish and english translation
http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip

### References

This project is built upon the following resources, which were instrumental in understanding and implementing the transformer architecture:

1. Attention Is All You Need paper by Ashish Vaswani et al. (https://arxiv.org/pdf/1706.03762)

This seminal paper lays the foundation for the transformer architecture, introducing the core concepts of self-attention, encoder-decoder structure, and multi-head attention.
Key takeaways:

Transformers rely solely on attention mechanisms for dependency modeling, eliminating the need for recurrent or convolutional layers.

Self-attention allows models to capture long-range dependencies and contextual information effectively.

The encoder-decoder structure is well-suited for sequence-to-sequence tasks.

2. The Illustrated Transformer by Jay Alammar (http://jalammar.github.io/illustrated-transformer/)

This interactive visualization tool provides a comprehensive and intuitive understanding of the transformer's inner workings.

Benefits:

Step-by-step visualization of transformer layers and processes.

Interactive exploration of attention matrices, aiding in understanding how the model focuses on relevant parts of the input.

A valuable resource for both beginners and experienced individuals seeking to deepen their knowledge.

3. Code Emporium Transformer YouTube Playlist (https://www.youtube.com/@CodeEmporium)

This video series offers a clear and detailed code walkthrough, explaining the implementation of the transformer architecture from scratch.

Advantages:

Practical, hands-on approach to transformer implementation.

Code snippets and explanations to enhance understanding.

Reinforces the theoretical concepts with practical implementation examples.
