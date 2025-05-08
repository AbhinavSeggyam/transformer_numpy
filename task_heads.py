import numpy as np

class TaskHead:
    """
    Base class for all task-specific heads.
    Defines the interface for forward pass, backward pass, and state management.
    """
    def __init__(self, d_model):
        self.d_model = d_model
        self.params = {}
        self.grads = {}
    
    def __call__(self, x):
        raise NotImplementedError
    
    def backward(self, d_out):
        raise NotImplementedError
    
    def get_state(self):
        return self.params
    
    def set_state(self, state):
        self.params = state

class ClassificationHead(TaskHead):
    """
    Head for classification tasks.
    Uses mean pooling over sequence length and projects to number of classes.
    """
    def __init__(self, d_model, num_classes):
        super().__init__(d_model)
        self.num_classes = num_classes
        
        # Initialize classification layer
        scale = np.sqrt(2.0 / (d_model + num_classes))
        self.W = np.random.randn(d_model, num_classes) * scale
        self.b = np.zeros(num_classes)
        
        self.params = {
            'W': self.W,
            'b': self.b
        }
        
        self.grads = {
            'W': np.zeros_like(self.W),
            'b': np.zeros_like(self.b)
        }
    
    def __call__(self, x):
        """Forward pass: mean pooling + linear projection"""
        # For classification, we typically use the [CLS] token or mean pooling
        if len(x.shape) == 3:  # [batch, seq_len, d_model]
            x = np.mean(x, axis=1)  # Mean pooling
        
        # Linear projection
        logits = np.matmul(x, self.W) + self.b
        return logits
    
    def backward(self, d_out):
        """Backward pass: compute gradients for weights and input"""
        # Gradient w.r.t. weights and bias
        self.grads['W'] = np.matmul(self.x.T, d_out)
        self.grads['b'] = np.sum(d_out, axis=0)
        
        # Gradient w.r.t. input
        d_x = np.matmul(d_out, self.W.T)
        return d_x

class SequenceLabelingHead(TaskHead):
    """
    Head for sequence labeling tasks (e.g., NER, POS tagging).
    Projects each token to a label distribution.
    """
    def __init__(self, d_model, num_labels):
        super().__init__(d_model)
        self.num_labels = num_labels
        
        # Initialize labeling layer
        scale = np.sqrt(2.0 / (d_model + num_labels))
        self.W = np.random.randn(d_model, num_labels) * scale
        self.b = np.zeros(num_labels)
        
        self.params = {
            'W': self.W,
            'b': self.b
        }
        
        self.grads = {
            'W': np.zeros_like(self.W),
            'b': np.zeros_like(self.b)
        }
    
    def __call__(self, x):
        """Forward pass: project each token to label space"""
        # Linear projection for each token
        logits = np.matmul(x, self.W) + self.b
        return logits
    
    def backward(self, d_out):
        """Backward pass: compute gradients for weights and input"""
        # Gradient w.r.t. weights and bias
        self.grads['W'] = np.matmul(self.x.reshape(-1, self.d_model).T, 
                                   d_out.reshape(-1, self.num_labels))
        self.grads['b'] = np.sum(d_out, axis=(0, 1))
        
        # Gradient w.r.t. input
        d_x = np.matmul(d_out, self.W.T)
        return d_x

class GenerationHead(TaskHead):
    """
    Head for text generation tasks.
    Projects each position to vocabulary distribution.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__(d_model)
        self.vocab_size = vocab_size
        
        # Initialize generation layer
        scale = np.sqrt(2.0 / (d_model + vocab_size))
        self.W = np.random.randn(d_model, vocab_size) * scale
        self.b = np.zeros(vocab_size)
        
        self.params = {
            'W': self.W,
            'b': self.b
        }
        
        self.grads = {
            'W': np.zeros_like(self.W),
            'b': np.zeros_like(self.b)
        }
    
    def __call__(self, x):
        """Forward pass: project each position to vocabulary"""
        # Linear projection to vocabulary size
        logits = np.matmul(x, self.W) + self.b
        return logits
    
    def backward(self, d_out):
        """Backward pass: compute gradients for weights and input"""
        # Gradient w.r.t. weights and bias
        self.grads['W'] = np.matmul(self.x.reshape(-1, self.d_model).T, 
                                   d_out.reshape(-1, self.vocab_size))
        self.grads['b'] = np.sum(d_out, axis=(0, 1))
        
        # Gradient w.r.t. input
        d_x = np.matmul(d_out, self.W.T)
        return d_x

class QuestionAnsweringHead(TaskHead):
    """
    Head for question answering tasks.
    Predicts start and end positions of answers in context.
    """
    def __init__(self, d_model):
        super().__init__(d_model)
        
        # Initialize start and end position layers
        scale = np.sqrt(2.0 / d_model)
        self.W_start = np.random.randn(d_model, 1) * scale
        self.W_end = np.random.randn(d_model, 1) * scale
        
        self.params = {
            'W_start': self.W_start,
            'W_end': self.W_end
        }
        
        self.grads = {
            'W_start': np.zeros_like(self.W_start),
            'W_end': np.zeros_like(self.W_end)
        }
    
    def __call__(self, x):
        """Forward pass: predict start and end positions"""
        # Project to start and end positions
        start_logits = np.matmul(x, self.W_start).squeeze(-1)
        end_logits = np.matmul(x, self.W_end).squeeze(-1)
        return start_logits, end_logits
    
    def backward(self, d_start, d_end):
        """Backward pass: compute gradients for weights and input"""
        # Gradient w.r.t. weights
        self.grads['W_start'] = np.matmul(self.x.reshape(-1, self.d_model).T, 
                                         d_start.reshape(-1, 1))
        self.grads['W_end'] = np.matmul(self.x.reshape(-1, self.d_model).T, 
                                       d_end.reshape(-1, 1))
        
        # Gradient w.r.t. input
        d_x = (np.matmul(d_start.reshape(-1, 1), self.W_start.T) +
               np.matmul(d_end.reshape(-1, 1), self.W_end.T))
        return d_x.reshape(self.x.shape) 