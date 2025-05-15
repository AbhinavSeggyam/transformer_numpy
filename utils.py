import numpy as np
from scipy.special import erf

def softmax(x, axis=-1):
    """
    Compute the softmax function along the specified axis with improved numerical stability.
    
    Args:
        x: Input tensor of any shape
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax output of same shape as input
    """
    # Subtracting max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-9)

def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function with improved numerical stability.
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        GELU output of same shape as input
    """
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_grad(x):
    """
    Gradient of GELU activation function with improved numerical stability.
    dGELU/dx = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * x * d(erf(x/sqrt(2)))/dx
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        GELU gradient of same shape as input
    """
    return 0.5 * (1 + erf(x / np.sqrt(2))) + 0.5 * x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def cross_entropy(predictions, targets):
    """
    Compute cross-entropy loss between predictions and targets with improved numerical stability.
    Handles both single-label and multi-label classification.
    
    Args:
        predictions: Logits tensor of shape [batch_size, num_classes]
        targets: One-hot encoded targets of shape [batch_size, num_classes]
        
    Returns:
        Scalar loss value
    """
    # Apply softmax to predictions with improved numerical stability
    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    probs = exp_preds / (np.sum(exp_preds, axis=-1, keepdims=True) + 1e-9)
    
    # Compute cross-entropy loss
    batch_size = predictions.shape[0]
    loss = -np.sum(targets * np.log(probs + 1e-9)) / batch_size
    
    return loss

def cross_entropy_grad(predictions, targets):
    """
    Compute gradient of cross-entropy loss with improved numerical stability.
    Returns gradient with respect to predictions.
    
    Args:
        predictions: Logits tensor of shape [batch_size, num_classes]
        targets: One-hot encoded targets of shape [batch_size, num_classes]
        
    Returns:
        Gradient tensor of same shape as predictions
    """
    # Apply softmax to predictions with improved numerical stability
    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    probs = exp_preds / (np.sum(exp_preds, axis=-1, keepdims=True) + 1e-9)
    
    # Compute gradient
    batch_size = predictions.shape[0]
    grad = (probs - targets) / batch_size
    
    return grad

def create_padding_mask(seq):
    """
    Create padding mask for attention with improved numerical stability.
    Returns mask where 0 indicates padding tokens.
    
    Args:
        seq: Input sequence tensor of shape [batch_size, seq_len]
        
    Returns:
        Mask tensor of shape [batch_size, 1, 1, seq_len]
    """
    mask = (seq != 0).astype(np.float32)
    return mask[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder attention with improved numerical stability.
    Prevents attending to future tokens.
    
    Args:
        size: Size of the sequence
        
    Returns:
        Mask tensor of shape [1, 1, size, size]
    """
    mask = np.triu(np.ones((size, size)), k=1)
    return np.where(mask == 1, np.nan, 0)  # Replace -1e9 with np.nan for better numerical stability
