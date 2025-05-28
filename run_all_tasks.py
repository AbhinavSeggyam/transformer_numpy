import numpy as np
from config import TransformerConfig
from train import Trainer
from utils import create_padding_mask, create_look_ahead_mask

# Helper to create sample data and vocab
vocab = ['<pad>', '<unk>', '<sos>', '<eos>', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'in', 'park']
word2idx = {word: idx for idx, word in enumerate(vocab)}

def create_sample_data():
    sentences = [
        'the cat sat on the mat',
        'the dog ran in the park',
        'the cat ran in the park',
        'the dog sat on the mat'
    ]
    tokenized = []
    for sent in sentences:
        tokens = ['<sos>'] + sent.split() + ['<eos>']
        indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
        tokenized.append(indices)
    max_len = max(len(seq) for seq in tokenized)
    padded = [seq + [word2idx['<pad>']] * (max_len - len(seq)) for seq in tokenized]
    return np.array(padded)

def train_classification():
    print("\n=== Classification Task ===")
    X = create_sample_data()
    y = np.array([0, 1, 0, 1]).reshape(-1, 1)
    config = TransformerConfig(
        vocab_size=len(vocab), d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=X.shape[1], num_classes=2
    )
    trainer = Trainer(config, task_type='classification')
    train_data = [(X, y)] * 10
    val_data = [(X, y)] * 2
    trainer.train(train_data, val_data, num_epochs=2)
    print("Classification training done.")

def train_sequence_labeling():
    print("\n=== Sequence Labeling Task ===")
    X = create_sample_data()
    y = np.random.randint(0, 3, size=X.shape)
    config = TransformerConfig(
        vocab_size=len(vocab), d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=X.shape[1], num_labels=3
    )
    trainer = Trainer(config, task_type='sequence_labeling')
    train_data = [(X, y)] * 10
    val_data = [(X, y)] * 2
    trainer.train(train_data, val_data, num_epochs=2)
    print("Sequence labeling training done.")

def train_generation():
    print("\n=== Generation Task ===")
    X = create_sample_data()
    y = np.roll(X, -1, axis=1)
    y[:, -1] = word2idx['<pad>']
    config = TransformerConfig(
        vocab_size=len(vocab), d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=X.shape[1]
    )
    trainer = Trainer(config, task_type='generation')
    train_data = [(X, y)] * 10
    val_data = [(X, y)] * 2
    trainer.train(train_data, val_data, num_epochs=2)
    print("Generation training done.")

def train_question_answering():
    print("\n=== Question Answering Task ===")
    X = create_sample_data()
    start_pos = np.array([2, 2, 2, 2]).reshape(-1, 1)
    end_pos = np.array([4, 4, 4, 4]).reshape(-1, 1)
    config = TransformerConfig(
        vocab_size=len(vocab), d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=X.shape[1]
    )
    trainer = Trainer(config, task_type='question_answering')
    train_data = [(X, start_pos)] * 10
    val_data = [(X, start_pos)] * 2
    trainer.train(train_data, val_data, num_epochs=2)
    print("Question answering training done.")

def main():
    train_classification()
    train_sequence_labeling()
    train_generation()
    train_question_answering()
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main() 