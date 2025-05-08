from attention import MultiHeadAttention
from layers import LayerNorm, FeedForward
import numpy as np  # Added explicit import

class DecoderLayer:
    """
    A single layer of the decoder, consisting of:
    1. Masked self-attention (prevents attending to future positions)
    2. Encoder-decoder attention (attends to encoder's output)
    3. Position-wise feed-forward network
    Each sub-layer is followed by layer normalization and residual connection.
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Initialize attention and feed-forward
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Initialize layer normalization
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        
        # Initialize parameters
        self.params = {
            'self_attention': self.self_attention.params,
            'enc_dec_attention': self.enc_dec_attention.params,
            'feed_forward': self.feed_forward.params,
            'layer_norm1': self.layer_norm1.params,
            'layer_norm2': self.layer_norm2.params,
            'layer_norm3': self.layer_norm3.params
        }
        
        # Initialize gradients
        self.grads = {
            'self_attention': np.zeros_like(self.self_attention.params),
            'enc_dec_attention': np.zeros_like(self.enc_dec_attention.params),
            'feed_forward': np.zeros_like(self.feed_forward.params),
            'layer_norm1': np.zeros_like(self.layer_norm1.params),
            'layer_norm2': np.zeros_like(self.layer_norm2.params),
            'layer_norm3': np.zeros_like(self.layer_norm3.params)
        }

    def __call__(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through the decoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, src_seq_len, d_model]
            src_mask: Mask for encoder-decoder attention
            tgt_mask: Mask for self-attention (usually causal mask)
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Store inputs for backward pass
        self.input = x
        self.enc_output = enc_output
        
        # Step 1: Masked self-attention
        self.attn_out = self.self_attention(x, x, x, tgt_mask)
        self.x1 = self.layer_norm1(x + self.attn_out)  # Add & Norm
        
        # Step 2: Encoder-decoder attention
        self.enc_dec_out = self.enc_dec_attention(self.x1, enc_output, enc_output, src_mask)
        self.x2 = self.layer_norm2(self.x1 + self.enc_dec_out)  # Add & Norm
        
        # Step 3: Feed-forward network
        self.ff_out = self.feed_forward(self.x2)
        self.output = self.layer_norm3(self.x2 + self.ff_out)  # Add & Norm
        
        return self.output

    def backward(self, d_output):
        """
        Backward pass through the decoder layer.
        
        Args:
            d_output: Gradient of the output
            
        Returns:
            Gradient of the input
        """
        # Step 1: Backward through final layer norm
        d_norm3, d_gamma3, d_beta3 = self.layer_norm3.backward(d_output)
        
        # Step 2: Backward through feed-forward
        d_ff = d_norm3 + d_ff
        d_ff, dW1, db1, dW2, db2 = self.feed_forward.backward(d_ff)
        
        # Step 3: Backward through second layer norm and residual
        d_x2 = d_ff
        d_norm2, d_gamma2, d_beta2 = self.layer_norm2.backward(d_x2)
        
        # Step 4: Backward through encoder-decoder attention
        d_enc_dec, dW_q_enc, dW_k_enc, dW_v_enc, dW_o_enc = self.enc_dec_attention.backward(d_norm2)
        
        # Step 5: Backward through first layer norm and residual
        d_x1 = d_norm2 + d_enc_dec
        d_norm1, d_gamma1, d_beta1 = self.layer_norm1.backward(d_x1)
        
        # Step 6: Backward through self-attention
        d_self_attn, dW_q_self, dW_k_self, dW_v_self, dW_o_self = self.self_attention.backward(d_norm1)
        
        # Step 7: Combine gradients for input
        d_input = d_norm1 + d_self_attn
        
        # Update layer gradients
        self.grads['feed_forward'] = d_ff
        self.grads['enc_dec_attention'] = d_enc_dec
        self.grads['self_attention'] = d_self_attn
        self.grads['layer_norm1'] = d_norm1
        self.grads['layer_norm2'] = d_norm2
        self.grads['layer_norm3'] = d_norm3
        
        return d_input
    
    def get_state(self):
        """
        Get the current state of the decoder layer for checkpointing.
        
        Returns:
            dict: Layer state including all parameters
        """
        return {
            'self_attention': self.self_attention.get_state(),
            'enc_dec_attention': self.enc_dec_attention.get_state(),
            'feed_forward': self.feed_forward.get_state(),
            'layer_norm1': self.layer_norm1.get_state(),
            'layer_norm2': self.layer_norm2.get_state(),
            'layer_norm3': self.layer_norm3.get_state()
        }

    def set_state(self, state):
        """Set state of decoder layer"""
        self.self_attention.set_state(state['self_attention'])
        self.enc_dec_attention.set_state(state['enc_dec_attention'])
        self.feed_forward.set_state(state['feed_forward'])
        self.layer_norm1.set_state(state['layer_norm1'])
        self.layer_norm2.set_state(state['layer_norm2'])
        self.layer_norm3.set_state(state['layer_norm3'])

class Decoder:
    """
    The decoder consists of a stack of identical decoder layers.
    Each layer processes the input through self-attention, encoder-decoder attention,
    and a feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Initialize decoder layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, num_heads, d_ff))
        
        # Initialize parameters
        self.params = {
            'layers': [layer.params for layer in self.layers]
        }
        
        # Initialize gradients
        self.grads = {
            'layers': [np.zeros_like(layer.params) for layer in self.layers]
        }

    def __call__(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output of shape [batch_size, src_seq_len, d_model]
            src_mask: Mask for encoder-decoder attention
            tgt_mask: Mask for self-attention (usually causal mask)
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def backward(self, d_output):
        """
        Backward pass through all decoder layers.
        
        Args:
            d_output: Gradient of the output
            
        Returns:
            List of gradients for all layers
        """
        d_x = d_output
        layer_grads = []
        
        for layer in reversed(self.layers):
            d_x = layer.backward(d_x)
            layer_grads.append(layer.grads)
        
        self.grads['layers'] = list(reversed(layer_grads))
        return d_x

    def get_state(self):
        """
        Get the current state of the decoder for checkpointing.
        
        Returns:
            dict: Decoder state including all layer states
        """
        return {
            'layers': [layer.get_state() for layer in self.layers]
        }

    def set_state(self, state):
        """Set state of decoder"""
        for layer, layer_state in zip(self.layers, state['layers']):
            layer.set_state(layer_state)
