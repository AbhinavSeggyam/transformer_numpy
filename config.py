# Basic configuration class of a transformer
class TransformerConfig:
    def __init__(self, 
                 vocab_size=30000,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_layers=6,
                 max_len=512,
                 dropout=0.1,
                 num_classes=None,
                 num_labels=None,
                 learning_rate=1e-4,
                 warmup_steps=4000):
        """
        Initialize transformer configuration.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of the feed-forward network
            num_layers (int): Number of encoder/decoder layers
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
            num_classes (int): Number of classes for classification tasks
            num_labels (int): Number of labels for sequence labeling tasks
            learning_rate (float): Learning rate for optimizer
            warmup_steps (int): Warmup steps for learning rate scheduler
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
