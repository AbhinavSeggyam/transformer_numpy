import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.positional_encoding = pe[np.newaxis, :, :]
        
        # Store parameters (though positional encoding is fixed)
        self.params = {
            'positional_encoding': self.positional_encoding
        }
        
        # Initialize gradients (though they won't be used)
        self.grads = {
            'positional_encoding': np.zeros_like(self.positional_encoding)
        }
    
    def __call__(self, x):
        """Forward pass"""
        # Add positional encoding to input
        return x + self.positional_encoding[:, :x.shape[1], :]
    
    def backward(self, d_out):
        """Backward pass"""
        # Gradient just passes through
        return d_out
    
    def get_state(self):
        """Get current state of positional encoding"""
        return {
            'positional_encoding': self.positional_encoding.copy()
        }
    
    def set_state(self, state):
        """Set state of positional encoding"""
        self.positional_encoding = state['positional_encoding'].copy()
        
        # Update params dictionary
        self.params.update({
            'positional_encoding': self.positional_encoding
        })

class Embeddings:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embedding matrix with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.embedding = np.random.randn(vocab_size, d_model) * scale
        
        # Store parameters
        self.params = {
            'embedding': self.embedding
        }
        
        # Initialize gradients
        self.grads = {
            'embedding': np.zeros_like(self.embedding)
        }
    
    def __call__(self, x):
        """Forward pass"""
        # Store input for backward pass
        self.x = x
        
        # Get embeddings for input indices
        return self.embedding[x]
    
    def backward(self, d_out):
        """Backward pass"""
        # Initialize gradient matrix
        d_embedding = np.zeros_like(self.embedding)
        
        # Accumulate gradients for each input position
        np.add.at(d_embedding, self.x, d_out)
        
        # Store gradients
        self.grads['embedding'] = d_embedding
        
        return d_embedding
    
    def get_state(self):
        """Get current state of embeddings"""
        return {
            'embedding': self.embedding.copy()
        }
    
    def set_state(self, state):
        """Set state of embeddings"""
        self.embedding = state['embedding'].copy()
        
        # Update params dictionary
        self.params.update({
            'embedding': self.embedding
        })