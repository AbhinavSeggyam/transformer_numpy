from config import TransformerConfig
from embeddings import Embeddings, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from task_heads import TaskHead, ClassificationHead, SequenceLabelingHead, GenerationHead, QuestionAnsweringHead
from utils import cross_entropy, cross_entropy_grad
import numpy as np

class Transformer:
    """
    Main transformer model supporting multiple NLP tasks.
    Implements the encoder-decoder architecture with task-specific heads.
    """
    def __init__(self, config, task_type='generation'):
        # Store configuration and model dimensions
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.num_layers = config.num_layers
        self.max_len = config.max_len
        self.task_type = task_type
        
        # Initialize model components
        self.embedding = Embeddings(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_len)
        self.encoder = Encoder(self.d_model, self.num_heads, self.d_ff, self.num_layers)
        self.decoder = Decoder(self.d_model, self.num_heads, self.d_ff, self.num_layers)
        self.task_head = self._create_task_head(task_type)
        
        # Initialize parameter and gradient dictionaries
        self.params = {
            'embedding': self.embedding.params,
            'encoder': self.encoder.params,
            'decoder': self.decoder.params,
            'task_head': self.task_head.params
        }
        self.grads = {
            'embedding': np.zeros_like(self.embedding.params),
            'encoder': np.zeros_like(self.encoder.params),
            'decoder': np.zeros_like(self.decoder.params),
            'task_head': np.zeros_like(self.task_head.params)
        }
    
    def _create_task_head(self, task_type):
        """Create appropriate task head based on task type"""
        if task_type == 'classification':
            return ClassificationHead(self.d_model, self.config.num_classes)
        elif task_type == 'sequence_labeling':
            return SequenceLabelingHead(self.d_model, self.config.num_labels)
        elif task_type == 'generation':
            return GenerationHead(self.d_model, self.vocab_size)
        elif task_type == 'question_answering':
            return QuestionAnsweringHead(self.d_model)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def __call__(self, src, tgt=None, src_mask=None, tgt_mask=None):
        """
        Forward pass through the transformer.
        For generation tasks, uses both encoder and decoder.
        For other tasks, uses only encoder output.
        """
        # Store inputs for backward pass
        self.src = src
        self.tgt = tgt
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask
        
        # Process input through embeddings and encoder
        self.src_embedded = self.positional_encoding(self.embedding(src))
        self.enc_output = self.encoder(self.src_embedded, src_mask)
        
        if self.task_type == 'generation' and tgt is not None:
            # For generation tasks, use decoder
            self.tgt_embedded = self.positional_encoding(self.embedding(tgt))
            self.dec_output = self.decoder(self.tgt_embedded, self.enc_output, src_mask, tgt_mask)
            output = self.task_head(self.dec_output)
        else:
            # For other tasks, use encoder output directly
            output = self.task_head(self.enc_output)
        
        return output
    
    def compute_loss(self, predictions, targets):
        """Compute task-specific loss"""
        if self.task_type == 'classification':
            return self._compute_classification_loss(predictions, targets)
        elif self.task_type == 'sequence_labeling':
            return self._compute_sequence_labeling_loss(predictions, targets)
        elif self.task_type == 'generation':
            return self._compute_generation_loss(predictions, targets)
        elif self.task_type == 'question_answering':
            return self._compute_qa_loss(predictions, targets)
    
    def _compute_classification_loss(self, predictions, targets):
        """Compute cross-entropy loss for classification"""
        return cross_entropy(predictions, targets)
    
    def _compute_sequence_labeling_loss(self, predictions, targets):
        """Compute cross-entropy loss for sequence labeling"""
        batch, seq_len, num_labels = predictions.shape
        predictions = predictions.reshape(-1, num_labels)
        targets = targets.reshape(-1)
        return cross_entropy(predictions, targets)
    
    def _compute_generation_loss(self, predictions, targets):
        """Compute cross-entropy loss for generation"""
        batch, seq_len, vocab = predictions.shape
        predictions = predictions.reshape(-1, vocab)
        targets = targets.reshape(-1)
        return cross_entropy(predictions, targets)
    
    def _compute_qa_loss(self, predictions, targets):
        """Compute loss for question answering"""
        start_logits, end_logits = predictions
        start_pos, end_pos = targets
        
        # Compute cross-entropy loss for both start and end positions
        start_loss = cross_entropy(start_logits, start_pos)
        end_loss = cross_entropy(end_logits, end_pos)
        
        return start_loss + end_loss
    
    def backward(self, d_out):
        """
        Backward pass through the model.
        Propagates gradients through task head, decoder (if used), and encoder.
        """
        # Backward through task head
        if self.task_type == 'question_answering':
            d_start, d_end = d_out
            d_task = self.task_head.backward(d_start, d_end)
        else:
            d_task = self.task_head.backward(d_out)
        
        if self.task_type == 'generation' and self.tgt is not None:
            # Backward through decoder
            d_dec = self.decoder.backward(d_task, self.enc_output, self.src_mask, self.tgt_mask)
            d_enc = d_dec[0][5]  # Get d_enc_output from decoder grads
        else:
            d_enc = d_task
        
        # Backward through encoder and embeddings
        d_enc_output = self.encoder.backward(d_enc)
        d_src_embed = self.embedding.backward(self.positional_encoding.backward(d_enc_output))
        
        # Collect all gradients
        grads = {
            'task_head': self.task_head.grads,
            'embedding': d_src_embed,
            'encoder': self.encoder.grads,
            'decoder': self.decoder.grads if self.task_type == 'generation' else {}
        }
        
        return grads
    
    def get_state(self):
        """Get current state of all model components"""
        return {
            'embedding': self.embedding.get_state(),
            'encoder': self.encoder.get_state(),
            'decoder': self.decoder.get_state(),
            'task_head': self.task_head.get_state()
        }
    
    def set_state(self, state):
        """Set state of all model components"""
        self.embedding.set_state(state['embedding'])
        self.encoder.set_state(state['encoder'])
        self.decoder.set_state(state['decoder'])
        self.task_head.set_state(state['task_head'])
    