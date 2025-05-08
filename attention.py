from utils import softmax
import numpy as np

class MultiHeadAttention:
    """
    Multi-head attention mechanism as described in "Attention Is All You Need".
    This implementation allows for both self-attention and encoder-decoder attention.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention mechanism.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        # Initialize parameters
        self.params = {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        }
        
        # Initialize gradients
        self.grads = {
            'W_q': np.zeros_like(self.W_q),
            'W_k': np.zeros_like(self.W_k),
            'W_v': np.zeros_like(self.W_v),
            'W_o': np.zeros_like(self.W_o)
        }

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, d_k]
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine the heads back together.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, d_k]
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = x.shape[0]
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention with improved numerical stability.
        
        Args:
            Q: Query tensor of shape [batch_size, num_heads, seq_len, d_k]
            K: Key tensor of shape [batch_size, num_heads, seq_len, d_k]
            V: Value tensor of shape [batch_size, num_heads, seq_len, d_k]
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, d_k]
        """
        # Compute attention scores with scaling
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 3:  # [batch_size, seq_len, seq_len]
                mask = mask[:, None, :, :]  # Add head dimension
            scores = np.where(mask == 0, -1e9, scores)

        # Compute attention weights with improved numerical stability
        # Subtract max for numerical stability
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores = scores - scores_max
        exp_scores = np.exp(scores)
        
        # Apply mask to exp_scores if provided
        if mask is not None:
            exp_scores = np.where(mask == 0, 0, exp_scores)
        
        # Compute attention weights
        attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
        
        # Store attention weights for backward pass
        self.attn_weights = attn
        
        # Compute output
        output = np.matmul(attn, V)
        return output

    def causal_mask(self, seq_len):
        """
        Create a causal mask for decoder self-attention.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Binary mask tensor of shape [1, 1, seq_len, seq_len]
        """
        # Create a lower triangular matrix
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        # Add batch and head dimensions
        mask = mask[None, None, :, :]
        
        return mask.astype(np.float32)

    def __call__(self, Q, K, V, mask=None):
        """
        Forward pass through the multi-head attention mechanism.
        
        Args:
            Q: Query tensor of shape [batch_size, seq_len, d_model]
            K: Key tensor of shape [batch_size, seq_len, d_model]
            V: Value tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor or "causal" for causal masking
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = Q.shape[0]

        # Linear projections with batch processing
        Q_proj = np.matmul(Q, self.W_q)  # (batch, seq_len, d_model)
        K_proj = np.matmul(K, self.W_k)
        V_proj = np.matmul(V, self.W_v)

        # Store inputs and projections for backward pass
        self.input_Q = Q
        self.input_K = K
        self.input_V = V
        self.Q_proj = Q_proj
        self.K_proj = K_proj
        self.V_proj = V_proj

        # Split into heads
        Q_split = self.split_heads(Q_proj)
        K_split = self.split_heads(K_proj)
        V_split = self.split_heads(V_proj)

        self.Q_split = Q_split
        self.K_split = K_split
        self.V_split = V_split

        # Apply causal mask if required
        if isinstance(mask, str) and mask == "causal":
            seq_len = Q.shape[1]
            mask = self.causal_mask(seq_len)

        # Scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask)
        self.attn_output = attn_output

        # Concatenate heads
        concat_output = self.combine_heads(attn_output)
        self.concat_output = concat_output

        # Final linear projection
        output = np.matmul(concat_output, self.W_o)
        self.output = output
        
        return output

    def backward(self, d_output):
        """
        Backward pass through the multi-head attention mechanism.
        
        Args:
            d_output: Gradient of the output tensor
            
        Returns:
            Tuple of (d_input, dW_q, dW_k, dW_v, dW_o)
        """
        # Gradient w.r.t. output projection
        dW_o = np.matmul(self.concat_output.reshape(-1, self.d_model).T, 
                        d_output.reshape(-1, self.d_model))
        d_concat_output = np.matmul(d_output, self.W_o.T)

        # Split heads back
        d_attn_output = self.split_heads(d_concat_output)

        # Gradient w.r.t. attention weights and values
        dV = np.matmul(self.attn_weights.transpose(0, 1, 3, 2), d_attn_output)
        d_attn_weights = np.matmul(d_attn_output, self.V_split.transpose(0, 1, 3, 2))

        # Gradient w.r.t. attention scores (softmax backward)
        d_scores = d_attn_weights * self.attn_weights * (1 - self.attn_weights)

        # Gradient w.r.t. queries and keys
        dQ_split = np.matmul(d_scores, self.K_split)
        dK_split = np.matmul(d_scores.transpose(0, 1, 3, 2), self.Q_split)

        # Merge heads
        dQ_proj = self.combine_heads(dQ_split)
        dK_proj = self.combine_heads(dK_split)
        dV_proj = self.combine_heads(dV)

        # Gradient w.r.t. projection weights
        dW_q = np.matmul(self.input_Q.reshape(-1, self.d_model).T, 
                        dQ_proj.reshape(-1, self.d_model))
        dW_k = np.matmul(self.input_K.reshape(-1, self.d_model).T, 
                        dK_proj.reshape(-1, self.d_model))
        dW_v = np.matmul(self.input_V.reshape(-1, self.d_model).T, 
                        dV_proj.reshape(-1, self.d_model))

        # Gradient w.r.t. inputs
        dQ = np.matmul(dQ_proj, self.W_q.T)
        dK = np.matmul(dK_proj, self.W_k.T)
        dV = np.matmul(dV_proj, self.W_v.T)

        # Combine gradients w.r.t. inputs
        d_input = dQ + dK + dV

        return d_input, dW_q, dW_k, dW_v, dW_o

    def get_state(self):
        """Get current state of attention mechanism"""
        return {
            'W_q': self.W_q.copy(),
            'W_k': self.W_k.copy(),
            'W_v': self.W_v.copy(),
            'W_o': self.W_o.copy()
        }

    def set_state(self, state):
        """Set state of attention mechanism"""
        self.W_q = state['W_q'].copy()
        self.W_k = state['W_k'].copy()
        self.W_v = state['W_v'].copy()
        self.W_o = state['W_o'].copy()
        
        # Update params dictionary
        self.grads.update({
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o
        })
