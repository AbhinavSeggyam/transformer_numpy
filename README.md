# NumPy Transformer Implementation

A pure NumPy implementation of the Transformer architecture that supports multiple NLP tasks including classification, sequence labeling, text generation, and question answering.

## Features

- Pure NumPy implementation without external deep learning frameworks
- Support for multiple NLP tasks:
  - Text Classification
  - Sequence Labeling (NER, POS tagging)
  - Text Generation
  - Question Answering
- Modular architecture with task-specific heads
- Complete forward and backward pass implementations
- Layer normalization and GELU activation
- Multi-head attention mechanism

## Project Structure

```
transformers_numpy/
├── transformer.py      # Main transformer model implementation
├── task_heads.py      # Task-specific heads for different NLP tasks
├── layers.py          # Core transformer layers (MultiHeadAttention, FeedForward, LayerNorm)
├── utils.py           # Utility functions (activation, loss, masking)
├── config.py          # Configuration class for model parameters
└── task_examples.py   # Examples demonstrating different tasks
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transformers_numpy.git
cd transformers_numpy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project includes example scripts demonstrating different tasks. Here's how to run them:

```python
from config import TransformerConfig
from transformer import Transformer

# Create model configuration
config = TransformerConfig(
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_len=512
)

# Initialize transformer for specific task
model = Transformer(config, task_type='classification')  # or 'sequence_labeling', 'generation', 'question_answering'

# Run task examples
python task_examples.py
```

## Supported Tasks

1. **Classification**
   - Binary and multi-class classification
   - Mean pooling over sequence length

2. **Sequence Labeling**
   - Token-level predictions
   - Support for NER, POS tagging

3. **Text Generation**
   - Next token prediction
   - Full encoder-decoder architecture

4. **Question Answering**
   - Start and end position prediction
   - Context-aware attention

## Implementation Details

- **Multi-Head Attention**: Implements scaled dot-product attention with multiple heads
- **Feed-Forward Network**: Two linear transformations with GELU activation
- **Layer Normalization**: Normalizes inputs across features
- **Positional Encoding**: Sinusoidal positional encodings
- **Task Heads**: Specialized output layers for different tasks

## Requirements

- Python 3.6+
- NumPy
- SciPy

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 