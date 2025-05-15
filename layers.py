from utils import gelu, gelu_grad
import numpy as np
from scipy.special import erf

class LayerNorm:
    """
    Layer normalization as described in "Attention Is All You Need".
    Normalizes the input across the last dimension and applies learnable scale and shift.
    """
    def __init__(self, d_model, eps=1e-12):
        """
        Initialize layer normalization.
        
        Args:
            d_model: Dimension of the model
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Initialize parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
        # Store parameters
        self.params = {
            'gamma': self.gamma,
            'beta': self.beta
        }
        
        # Initialize gradients
        self.grads = {
            'gamma': np.zeros_like(self.gamma),
            'beta': np.zeros_like(self.beta)
        }

    def __call__(self, x):
        """
        Forward pass through layer normalization with improved numerical stability.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Store input for backward pass
        self.x = x
        
        # Calculate mean and variance with improved numerical stability
        self.mean = np.mean(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        x_centered = x - self.mean  # [batch_size, seq_len, d_model]
        self.var = np.mean(x_centered * x_centered, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        
        # Normalize with improved numerical stability
        self.x_norm = x_centered / np.sqrt(self.var + self.eps)  # [batch_size, seq_len, d_model]
        
        # Scale and shift
        return self.gamma * self.x_norm + self.beta  # [batch_size, seq_len, d_model]

    def backward(self, d_out):
        """
        Backward pass through layer normalization with improved numerical stability.
        
        Args:
            d_out: Gradient of the output [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of (dx, dgamma, dbeta)
        """
        # Gradient w.r.t. gamma and beta
        self.grads['gamma'] = np.sum(d_out * self.x_norm, axis=(0, 1))  # [d_model]
        self.grads['beta'] = np.sum(d_out, axis=(0, 1))  # [d_model]
        
        # Gradient w.r.t. input
        N = d_out.shape[0] * d_out.shape[1]  # Total number of elements in batch and sequence
        x_centered = self.x - self.mean  # [batch_size, seq_len, d_model]
        std_inv = 1 / np.sqrt(self.var + self.eps)  # [batch_size, seq_len, 1]
        
        # Gradient w.r.t. normalized input
        d_x_norm = d_out * self.gamma  # [batch_size, seq_len, d_model]
        
        # Gradient w.r.t. variance
        d_var = -0.5 * np.sum(d_x_norm * x_centered, axis=-1, keepdims=True) * std_inv**3  # [batch_size, seq_len, 1]
        
        # Gradient w.r.t. mean
        d_mean = -np.sum(d_x_norm * std_inv, axis=-1, keepdims=True) - 2 * d_var * np.mean(x_centered, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        
        # Gradient w.r.t. input
        d_x = d_x_norm * std_inv + 2 * d_var * x_centered / N + d_mean / N  # [batch_size, seq_len, d_model]
        
        return d_x

    def get_state(self):
        """Get current state of layer normalization"""
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def set_state(self, state):
        """Set state of layer normalization"""
        self.gamma = state['gamma']
        self.beta = state['beta']

class FeedForward:
    """
    Position-wise feed-forward network as described in "Attention Is All You Need".
    Consists of two linear transformations with a GELU activation in between.
    """
    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network with proper weight initialization.
        
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward network
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights and biases
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
        
        # Store parameters
        self.params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        
        # Initialize gradients
        self.grads = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }

    def __call__(self, x):
        """
        Forward pass through feed-forward network with improved GELU implementation.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of same shape as input
        """
        # Store input for backward pass
        self.x = x  # [batch_size, seq_len, d_model]
        
        # First linear transformation with improved numerical stability
        self.hidden = np.matmul(x, self.W1) + self.b1  # [batch_size, seq_len, d_ff]
        
        # GELU activation with improved numerical stability
        self.hidden = gelu(self.hidden)  # [batch_size, seq_len, d_ff]
        
        # Second linear transformation
        self.output = np.matmul(self.hidden, self.W2) + self.b2  # [batch_size, seq_len, d_model]
        
        return self.output

    def backward(self, d_out):
        """
        Backward pass through feed-forward network with improved GELU gradient.
        
        Args:
            d_out: Gradient of the output [batch_size, seq_len, d_model]
            
        Returns:
            Tuple of (d_input, dW1, db1, dW2, db2)
        """
        # Gradient w.r.t. second linear transformation
        d_hidden = np.matmul(d_out, self.W2.T)  # [batch_size, seq_len, d_ff]
        self.grads['W2'] = np.matmul(self.hidden.reshape(-1, self.d_ff).T, 
                                    d_out.reshape(-1, self.d_model))  # [d_ff, d_model]
        self.grads['b2'] = np.sum(d_out, axis=(0, 1))  # [d_model]
        
        # Gradient w.r.t. GELU activation with improved numerical stability
        d_hidden_pre = d_hidden * gelu_grad(self.hidden)  # [batch_size, seq_len, d_ff]
        
        # Gradient w.r.t. first linear transformation
        self.grads['W1'] = np.matmul(self.x.reshape(-1, self.d_model).T, 
                                    d_hidden_pre.reshape(-1, self.d_ff))  # [d_model, d_ff]
        self.grads['b1'] = np.sum(d_hidden_pre, axis=(0, 1))  # [d_ff]
        
        # Gradient w.r.t. input
        d_x = np.matmul(d_hidden_pre, self.W1.T)  # [batch_size, seq_len, d_model]
        
        return d_x

    def get_state(self):
        """Get current state of feed-forward network"""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
    
    def set_state(self, state):
        """Set state of feed-forward network"""
        self.W1 = state['W1']
        self.b1 = state['b1']
        self.W2 = state['W2']
        self.b2 = state['b2']

class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    Splits input into multiple heads, computes attention for each, and combines results.
    """
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize attention weights
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        self.params = {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        }
        
        self.grads = {
            'W_q': np.zeros_like(self.W_q),
            'W_k': np.zeros_like(self.W_k),
            'W_v': np.zeros_like(self.W_v),
            'W_o': np.zeros_like(self.W_o)
        }
    
    def __call__(self, q, k, v, mask=None):
        """Forward pass: compute multi-head attention"""
        batch_size = q.shape[0]
        
        # Linear projections and reshape for multi-head
        q = np.matmul(q, self.W_q).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = np.matmul(k, self.W_k).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = np.matmul(v, self.W_v).reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attn_weights = self._softmax(scores)
        
        # Apply attention weights to values
        context = np.matmul(attn_weights, v)
        
        # Reshape and project to output
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.matmul(context, self.W_o)
        
        # Store intermediate values for backward pass
        self.q = q
        self.k = k
        self.v = v
        self.attn_weights = attn_weights
        self.context = context
        
        return output
    
    def backward(self, d_out):
        """Backward pass: compute gradients for attention weights and inputs"""
        batch_size = d_out.shape[0]
        
        # Gradient w.r.t. output projection
        d_context = np.matmul(d_out, self.W_o.T)
        d_W_o = np.matmul(self.context.reshape(-1, self.d_model).T, 
                         d_out.reshape(-1, self.d_model))
        
        # Reshape for multi-head
        d_context = d_context.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Gradient w.r.t. attention weights and values
        d_attn_weights = np.matmul(d_context, self.v.transpose(0, 1, 3, 2))
        d_v = np.matmul(self.attn_weights.transpose(0, 1, 3, 2), d_context)
        
        # Gradient w.r.t. attention scores
        d_scores = self._softmax_grad(self.attn_weights, d_attn_weights)
        
        # Gradient w.r.t. queries and keys
        d_q = np.matmul(d_scores, self.k)
        d_k = np.matmul(d_scores.transpose(0, 1, 3, 2), self.q)
        
        # Reshape and project gradients
        d_q = d_q.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Gradient w.r.t. weight matrices
        self.grads['W_q'] = np.matmul(self.q.reshape(-1, self.d_model).T, d_q.reshape(-1, self.d_model))
        self.grads['W_k'] = np.matmul(self.k.reshape(-1, self.d_model).T, d_k.reshape(-1, self.d_model))
        self.grads['W_v'] = np.matmul(self.v.reshape(-1, self.d_model).T, d_v.reshape(-1, self.d_model))
        self.grads['W_o'] = d_W_o
        
        return d_q, d_k, d_v
    
    def _softmax(self, x):
        """Compute softmax over the last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _softmax_grad(self, softmax_output, grad):
        """Compute gradient of softmax"""
        batch_size, num_heads, seq_len, _ = softmax_output.shape
        grad_reshaped = grad.reshape(-1, seq_len)
        softmax_reshaped = softmax_output.reshape(-1, seq_len)
        
        # Compute Jacobian matrix
        jacobian = np.zeros((batch_size * num_heads * seq_len, seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    jacobian[:, i, j] = softmax_reshaped[:, i] * (1 - softmax_reshaped[:, i])
                else:
                    jacobian[:, i, j] = -softmax_reshaped[:, i] * softmax_reshaped[:, j]
        
        # Compute gradient
        grad_reshaped = grad_reshaped.reshape(-1, 1, seq_len)
        grad_softmax = np.matmul(grad_reshaped, jacobian).reshape(batch_size, num_heads, seq_len, seq_len)
        
        return grad_softmax