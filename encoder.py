from attention import MultiHeadAttention
from layers import LayerNorm, FeedForward

class EncoderLayer:
    def __init__(self, config):
        # Multi-head self-attention mechanism
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads)
        
        # First layer normalization (applied after attention + residual)
        self.norm1 = LayerNorm(config.d_model)
        
        # Position-wise feedforward network
        self.ff = FeedForward(config.d_model, config.d_ff)
        
        # Second layer normalization (applied after feedforward + residual)
        self.norm2 = LayerNorm(config.d_model)

    def __call__(self, x, mask=None):
        # Apply multi-head self-attention; Q=K=V=x in encoder
        attn_out = self.self_attn(x, x, x, mask)

        # Add & Norm: residual connection + layer normalization
        x_res = x + attn_out
        x_norm1 = self.norm1(x_res)

        # Apply position-wise feedforward network
        ff_out = self.ff(x_norm1)

        # Add & Norm: another residual connection + layer normalization
        x_res2 = x_norm1 + ff_out
        x_norm2 = self.norm2(x_res2)
        
        return x_norm2

    def backward(self, d_output):
        # Backprop through second layer normalization (after feedforward)
        d_res2 = self.norm2.backward(d_output)
        
        # Backprop through feedforward network
        d_ff = self.ff.backward(d_res2)
        
        # Backprop through first layer normalization (after attention)
        d_res1 = self.norm1.backward(d_ff)
        
        # Backprop through self-attention
        d_input, dW_q, dW_k, dW_v, dW_o = self.self_attn.backward(d_res1)
        
        # Gradient w.r.t. the inputs (this will be passed to the previous layer)
        return d_input, dW_q, dW_k, dW_v, dW_o, d_ff, d_res2

class Encoder:
    def __init__(self, config):
        # Stack of identical encoder layers
        self.layers = [EncoderLayer(config) for _ in range(config.num_layers)]
        # Initialize params dictionary
        self.params = {}
        for i, layer in enumerate(self.layers):
            self.params[f'layer_{i}'] = {
                'self_attn': layer.self_attn.params,
                'norm1': layer.norm1.params,
                'ff': layer.ff.params,
                'norm2': layer.norm2.params
            }

    def __call__(self, x, mask=None):
        # Pass input through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def backward(self, d_output):
        # Backprop through each encoder layer
        d_input = d_output
        all_grads = []
        for layer in reversed(self.layers):
            d_input, dW_q, dW_k, dW_v, dW_o, d_ff, d_res2 = layer.backward(d_input)
            all_grads.append((dW_q, dW_k, dW_v, dW_o, d_ff, d_res2))
        
        # Return all gradients for the model parameters and activations
        return all_grads
