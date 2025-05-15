from config import TransformerConfig
from embeddings import Embeddings, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from utils import cross_entropy, cross_entropy_grad
import numpy as np

class Transformer:
    def __init__(self, config: TransformerConfig):
        # Token embeddings for source and target inputs
        self.src_embed = Embeddings(config.vocab_size, config.d_model)
        self.tgt_embed = Embeddings(config.vocab_size, config.d_model)
        
        # Shared positional encoding added to embeddings
        self.pos_enc = PositionalEncoding(config.d_model, config.max_len)
        
        # Transformer encoder and decoder stacks
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Final linear projection layer from d_model to vocab size
        self.final_linear = np.random.randn(config.d_model, config.vocab_size) * 0.01

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        # Apply embedding + positional encoding to source and target
        src = self.pos_enc(self.src_embed(src))  # [batch, src_len, d_model]
        tgt = self.pos_enc(self.tgt_embed(tgt))  # [batch, tgt_len, d_model]

        # Pass source through encoder
        enc_out = self.encoder(src, src_mask)  # [batch, src_len, d_model]

        # Pass target and encoder output through decoder
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)  # [batch, tgt_len, d_model]

        # Project decoder output to vocabulary logits
        return dec_out @ self.final_linear  # [batch, tgt_len, vocab_size]
    
    def compute_loss(self, logits, targets):
        # Flatten logits and targets for cross-entropy computation
        batch, seq_len, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
        targets = targets.reshape(-1)
        return cross_entropy(logits, targets)  # Scalar loss

    def compute_grad(self, logits, targets):
        # Compute gradient of the loss with respect to logits
        batch, seq_len, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
        targets = targets.reshape(-1)
        return cross_entropy_grad(logits, targets).reshape(batch, seq_len, vocab)
    
    
