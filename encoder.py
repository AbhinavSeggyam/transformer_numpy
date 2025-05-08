# encoder.py
from layers import LayerNorm, FeedForward
from attention import MultiHeadAttention
import numpy as np

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Initialize components
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Initialize parameters
        self.params = {
            'self_attn': self.self_attn.params,
            'norm1': self.norm1.params,
            'norm2': self.norm2.params
        }
        
        # Initialize gradients
        self.grads = {
            'self_attn': self.self_attn.grads,
            'norm1': self.norm1.grads,
            'norm2': self.norm2.grads
        }
    
    def __call__(self, x, mask=None):
        # Store input for backward pass
        self.x = x
        
        # Self-attention sublayer
        self.attn_output = self.self_attn(x, x, x, mask)
        self.norm1_input = x + self.attn_output
        self.norm1_output = self.norm1(self.norm1_input)
        
        # Feed-forward sublayer
        self.ff_output = self.feed_forward(self.norm1_output)
        self.norm2_input = self.norm1_output + self.ff_output
        self.norm2_output = self.norm2(self.norm2_input)
        
        return self.norm2_output
    
    def backward(self, d_out):
        # Gradient through second normalization
        d_norm2_input = self.norm2.backward(d_out)
        
        # Split gradient between residual and feed-forward
        d_ff_output = d_norm2_input
        d_norm1_output = d_norm2_input
        
        # Gradient through feed-forward
        d_norm1_output_ff = self.feed_forward.backward(d_ff_output)
        d_norm1_output += d_norm1_output_ff
        
        # Gradient through first normalization
        d_norm1_input = self.norm1.backward(d_norm1_output)
        
        # Split gradient between residual and attention
        d_attn_output = d_norm1_input
        d_x = d_norm1_input
        
        # Gradient through self-attention
        d_x_attn = self.self_attn.backward(d_attn_output)
        d_x += d_x_attn
        
        # Collect all gradients
        self.grads = {
            'self_attn': self.self_attn.grads,
            'norm1': self.norm1.grads,
            'norm2': self.norm2.grads
        }
        
        return d_x
    
    def get_state(self):
        return {
            'self_attn': self.self_attn.get_state(),
            'feed_forward': self.feed_forward.get_state(),
            'norm1': self.norm1.get_state(),
            'norm2': self.norm2.get_state()
        }
    
    def set_state(self, state):
        self.self_attn.set_state(state['self_attn'])
        self.feed_forward.set_state(state['feed_forward'])
        self.norm1.set_state(state['norm1'])
        self.norm2.set_state(state['norm2'])

class Encoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        
        # Initialize layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, d_ff))
        
        # Initialize parameters
        self.params = {}
        for i, layer in enumerate(self.layers):
            self.params[f'layer_{i}'] = layer.params
        
        # Initialize gradients
        self.grads = {}
        for i, layer in enumerate(self.layers):
            self.grads[f'layer_{i}'] = layer.grads
    
    def __call__(self, x, mask=None):
        # Store input for backward pass
        self.x = x
        self.mask = mask
        
        # Pass through each layer
        self.layer_outputs = []
        for layer in self.layers:
            x = layer(x, mask)
            self.layer_outputs.append(x)
        
        return x
    
    def backward(self, d_out):
        # Gradient through each layer in reverse
        d_x = d_out
        for i in range(len(self.layers) - 1, -1, -1):
            d_x = self.layers[i].backward(d_x)
            
            # Update gradients
            self.grads[f'layer_{i}'] = self.layers[i].grads
        
        return d_x
    
    def get_state(self):
        state = {}
        for i, layer in enumerate(self.layers):
            state[f'layer_{i}'] = layer.get_state()
        return state
    
    def set_state(self, state):
        for i, layer in enumerate(self.layers):
            layer.set_state(state[f'layer_{i}'])
