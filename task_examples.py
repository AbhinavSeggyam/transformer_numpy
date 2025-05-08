import numpy as np
from config import TransformerConfig
from transformer import Transformer
from utils import create_padding_mask, create_look_ahead_mask

def create_sample_data():
    """Create sample data for different tasks"""
    # Sample vocabulary
    vocab = ['<pad>', '<unk>', '<sos>', '<eos>', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'in', 'park']
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Sample sentences
    sentences = [
        'the cat sat on the mat',
        'the dog ran in the park',
        'the cat ran in the park',
        'the dog sat on the mat'
    ]
    
    # Convert sentences to token indices
    tokenized = []
    for sent in sentences:
        tokens = ['<sos>'] + sent.split() + ['<eos>']
        indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        tokenized.append(indices)
    
    # Pad sequences
    max_len = max(len(seq) for seq in tokenized)
    padded = [seq + [word2idx['<pad>']] * (max_len - len(seq)) for seq in tokenized]
    
    return np.array(padded), word2idx

def classification_example():
    """Example of using transformer for classification"""
    # Create config
    config = TransformerConfig(
        vocab_size=13,  # Size of our sample vocabulary
        d_model=64,     # Smaller model for example
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=10,
        num_classes=2   # Binary classification
    )
    
    # Create model
    model = Transformer(config, task_type='classification')
    
    # Create sample data
    X, word2idx = create_sample_data()
    
    # Create labels (0 for sentences with 'cat', 1 for sentences with 'dog')
    y = np.array([0, 1, 0, 1])
    
    # Create masks
    src_mask = create_padding_mask(X)
    
    # Forward pass
    predictions = model(X, src_mask=src_mask)
    
    print("\nClassification Example:")
    print("Input shape:", X.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:", predictions)

def sequence_labeling_example():
    """Example of using transformer for sequence labeling"""
    # Create config
    config = TransformerConfig(
        vocab_size=13,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=10,
        num_labels=3  # 3 possible labels per token
    )
    
    # Create model
    model = Transformer(config, task_type='sequence_labeling')
    
    # Create sample data
    X, word2idx = create_sample_data()
    
    # Create random labels for each token
    y = np.random.randint(0, 3, size=X.shape)
    
    # Create masks
    src_mask = create_padding_mask(X)
    
    # Forward pass
    predictions = model(X, src_mask=src_mask)
    
    print("\nSequence Labeling Example:")
    print("Input shape:", X.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:", predictions)

def generation_example():
    """Example of using transformer for text generation"""
    # Create config
    config = TransformerConfig(
        vocab_size=13,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=10
    )
    
    # Create model
    model = Transformer(config, task_type='generation')
    
    # Create sample data
    X, word2idx = create_sample_data()
    
    # Create target sequence (shifted by one position)
    y = np.roll(X, -1, axis=1)
    y[:, -1] = word2idx['<pad>']
    
    # Create masks
    src_mask = create_padding_mask(X)
    tgt_mask = create_look_ahead_mask(y.shape[1])
    
    # Forward pass
    predictions = model(X, y, src_mask=src_mask, tgt_mask=tgt_mask)
    
    print("\nGeneration Example:")
    print("Input shape:", X.shape)
    print("Target shape:", y.shape)
    print("Output shape:", predictions.shape)
    print("Predictions:", predictions)

def question_answering_example():
    """Example of using transformer for question answering"""
    # Create config
    config = TransformerConfig(
        vocab_size=13,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=10
    )
    
    # Create model
    model = Transformer(config, task_type='question_answering')
    
    # Create sample data
    X, word2idx = create_sample_data()
    
    # Create start and end positions for answers
    start_pos = np.array([2, 2, 2, 2])  # Start at 'sat' or 'ran'
    end_pos = np.array([4, 4, 4, 4])    # End at 'mat' or 'park'
    
    # Create masks
    src_mask = create_padding_mask(X)
    
    # Forward pass
    predictions = model(X, src_mask=src_mask)
    
    print("\nQuestion Answering Example:")
    print("Input shape:", X.shape)
    print("Start positions:", start_pos)
    print("End positions:", end_pos)
    print("Predictions:", predictions)

def main():
    """Run all examples"""
    print("Running transformer examples for different tasks...")
    
    # Run classification example
    classification_example()
    
    # Run sequence labeling example
    sequence_labeling_example()
    
    # Run generation example
    generation_example()
    
    # Run question answering example
    question_answering_example()

if __name__ == "__main__":
    main() 