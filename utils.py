import numpy as np
from scipy.special import erf

def softmax(x, axis=-1):
    # Compute the softmax function along the specified axis
    # Subtracting max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_grad(x):
    """
    Gradient of GELU activation function.
    dGELU/dx = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * d(erf(x/sqrt(2)))/dx
    """
    return 0.5 * (1 + erf(x / np.sqrt(2))) + 0.5 * x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def cross_entropy(predictions, targets):
    """
    Compute cross-entropy loss between predictions and targets.
    Handles both single-label and multi-label classification.
    """
    # Apply softmax to predictions
    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    
    # Compute cross-entropy loss
    batch_size = predictions.shape[0]
    loss = -np.sum(targets * np.log(probs + 1e-10)) / batch_size
    
    return loss

def cross_entropy_grad(predictions, targets):
    """
    Compute gradient of cross-entropy loss.
    Returns gradient with respect to predictions.
    """
    # Apply softmax to predictions
    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    
    # Compute gradient
    batch_size = predictions.shape[0]
    grad = (probs - targets) / batch_size
    
    return grad

def create_padding_mask(seq):
    """
    Create padding mask for attention.
    Returns mask where 0 indicates padding tokens.
    """
    mask = (seq != 0).astype(np.float32)
    return mask[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder attention.
    Prevents attending to future tokens.
    """
    mask = np.triu(np.ones((size, size)), k=1)
    return mask * -1e9
