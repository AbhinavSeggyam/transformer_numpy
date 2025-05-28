import numpy as np
from config import TransformerConfig
from train import Trainer

def create_dummy_data(batch_size=32, seq_len=20, vocab_size=1000):
    """Create dummy training data"""
    src_data = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
    tgt_data = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
    return src_data, tgt_data

def main():
    # Create configuration
    config = TransformerConfig(
        vocab_size=1000,
        d_model=64,  # Smaller for testing
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=20,
        dropout=0.1
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Create dummy datasets
    train_data = [create_dummy_data() for _ in range(10)]  # 10 batches
    val_data = [create_dummy_data() for _ in range(2)]     # 2 batches
    
    # Train for 2 epochs
    print("Starting training...")
    trainer.train(train_data, val_data, num_epochs=2)
    print("Training completed!")

if __name__ == "__main__":
    main() 