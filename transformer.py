from config import TransformerConfig
from embeddings import Embeddings, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from utils import cross_entropy, cross_entropy_grad
import numpy as np

class Transformer:
    def __init__(self, config: TransformerConfig, task_type='generation'):
        # Token embeddings for source and target inputs
        self.src_embed = Embeddings(config.vocab_size, config.d_model)
        self.tgt_embed = Embeddings(config.vocab_size, config.d_model)
        
        # Shared positional encoding added to embeddings
        self.pos_enc = PositionalEncoding(config.d_model, config.max_len)
        
        # Transformer encoder and decoder stacks
        self.encoder = Encoder(config)
        self.decoder = Decoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers
        )
        
        # Final linear projection layer from d_model to vocab size
        self.final_linear = np.random.randn(config.d_model, config.vocab_size) * 0.01
        
        # Store task type and learning rate
        self.task_type = task_type
        self.learning_rate = config.learning_rate
        
        # Initialize grads dictionary
        self.grads = {}

    def __call__(self, src, tgt=None, src_mask=None, tgt_mask=None):
        # Apply embedding + positional encoding to source
        src_emb = self.pos_enc(self.src_embed(src))  # [batch, src_len, d_model]
        enc_out = self.encoder(src_emb, src_mask)  # [batch, src_len, d_model]
        
        # Store encoder output for backpropagation
        self.last_encoder_output = enc_out

        if self.task_type in ['classification', 'question_answering']:
            # For classification/QA, use only encoder output
            # Pool (mean) over sequence dimension, then project
            pooled = np.mean(enc_out, axis=1)  # [batch, d_model]
            self.last_pooled = pooled
            logits = pooled @ self.final_linear  # [batch, vocab_size or num_classes]
            self.last_logits = logits
            return logits
        else:
            # For generation/seq labeling, use decoder
            tgt_emb = self.pos_enc(self.tgt_embed(tgt))  # [batch, tgt_len, d_model]
            dec_out = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)  # [batch, tgt_len, d_model]
            self.last_decoder_output = dec_out
            logits = dec_out @ self.final_linear  # [batch, tgt_len, vocab_size]
            self.last_logits = logits
            return logits
    
    def compute_loss(self, logits, targets):
        # Handle both 2D (classification) and 3D (generation) logits
        if logits.ndim == 3:
            batch, seq_len, vocab = logits.shape
            logits_flat = logits.reshape(-1, vocab)
            targets_flat = targets.reshape(-1)
        else:  # 2D
            logits_flat = logits
            targets_flat = targets
        return cross_entropy(logits_flat, targets_flat)  # Scalar loss

    def compute_grad(self, logits, targets):
        # Handle both 2D (classification) and 3D (generation) logits
        if logits.ndim == 3:
            batch, seq_len, vocab = logits.shape
            logits_flat = logits.reshape(-1, vocab)
            targets_flat = targets.reshape(-1)
            grad = cross_entropy_grad(logits_flat, targets_flat).reshape(batch, seq_len, vocab)
        else:
            grad = cross_entropy_grad(logits, targets)
        return grad
    
    def backward(self, logits, targets):
        """Backward pass through the transformer"""
        # Get gradients from loss
        grad = self.compute_grad(logits, targets)
        
        if self.task_type in ['classification', 'question_answering']:
            # For classification/QA, grad shape: [batch, vocab_size or num_classes]
            grad_final = grad @ self.final_linear.T  # [batch, d_model]
            self.grads['final_linear'] = self.last_pooled.T @ grad
            self.final_linear -= self.learning_rate * self.grads['final_linear']
            # Backprop through encoder (pooled mean)
            encoder_grad = np.repeat(grad_final[:, np.newaxis, :], self.last_encoder_output.shape[1], axis=1) / self.last_encoder_output.shape[1]
            self.encoder.backward(encoder_grad)
        else:
            # For generation/seq labeling
            grad_final = grad @ self.final_linear.T
            self.grads['final_linear'] = self.last_decoder_output.T @ grad
            self.final_linear -= self.learning_rate * self.grads['final_linear']
            # Backpropagate through decoder
            decoder_grad = self.decoder.backward(grad_final, self.last_encoder_output)
            # Backpropagate through encoder
            self.encoder.backward(decoder_grad)
        # Store parameters for optimizer
        self.params = {
            'final_linear': self.final_linear,
            'encoder': self.encoder.params,
            'decoder': self.decoder.params,
            'src_embed': self.src_embed.params,
            'tgt_embed': self.tgt_embed.params
        }
    
    def get_state(self):
        """Get model state for checkpointing"""
        return {
            'final_linear': self.final_linear,
            'encoder': self.encoder.get_state(),
            'decoder': self.decoder.get_state(),
            'src_embed': self.src_embed.get_state(),
            'tgt_embed': self.tgt_embed.get_state()
        }
    
    def set_state(self, state):
        """Set model state from checkpoint"""
        self.final_linear = state['final_linear']
        self.encoder.set_state(state['encoder'])
        self.decoder.set_state(state['decoder'])
        self.src_embed.set_state(state['src_embed'])
        self.tgt_embed.set_state(state['tgt_embed'])
    
    
