# Transformer Data Flow Documentation

## Overview
This document provides a detailed explanation of how data flows through the transformer implementation for different tasks: classification, sequence labeling, generation, and question answering.

## Common Components

### 1. Data Preparation
```python
Input: Raw text sentences
Output: Tokenized and padded sequences
Flow:
1. Tokenize sentences using word2idx
2. Add <sos> and <eos> tokens
3. Pad sequences to max length
Shape: [batch_size, max_seq_len]
```

### 2. Model Architecture
```python
Components:
- Source Embeddings: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
- Positional Encoding: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
- Encoder: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
- Decoder (for generation/seq labeling): [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
- Final Linear Layer: [batch_size, d_model] -> [batch_size, vocab_size/num_classes]
```

## Task-Specific Data Flows

### 1. Classification Task

#### Forward Pass
```python
Input: src_batch [batch_size, seq_len], src_mask
Flow:
1. Source Embeddings:
   - Input: src_batch [batch_size, seq_len]
   - Output: [batch_size, seq_len, d_model]

2. Positional Encoding:
   - Input: embeddings [batch_size, seq_len, d_model]
   - Output: [batch_size, seq_len, d_model]

3. Encoder:
   - Input: encoded [batch_size, seq_len, d_model]
   - Process: Multiple encoder layers
   - Output: [batch_size, seq_len, d_model]

4. Pooling:
   - Input: encoder output [batch_size, seq_len, d_model]
   - Process: Mean pooling over sequence dimension
   - Output: [batch_size, d_model]

5. Final Linear Layer:
   - Input: pooled [batch_size, d_model]
   - Output: logits [batch_size, num_classes]

6. Loss Computation:
   - Input: logits [batch_size, num_classes], targets [batch_size, 1]
   - Process: Cross-entropy loss
   - Output: scalar loss
```

#### Backward Pass
```python
Input: loss, logits, targets
Flow:
1. Gradient Computation:
   - Input: logits [batch_size, num_classes], targets [batch_size, 1]
   - Output: grad [batch_size, num_classes]

2. Final Linear Layer:
   - grad_final = grad @ final_linear.T
   - grads['final_linear'] = last_pooled.T @ grad

3. Encoder:
   - encoder_grad = repeat(grad_final) / seq_len
   - Backprop through encoder layers

4. Parameter Updates:
   - Update using Adam optimizer
```

### 2. Sequence Labeling Task

#### Forward Pass
```python
Input: src_batch [batch_size, seq_len], tgt_batch [batch_size, seq_len], masks
Flow:
1. Source Processing:
   - Source embeddings + positional encoding
   - Encoder processing
   - Output: [batch_size, seq_len, d_model]

2. Target Processing:
   - Target embeddings + positional encoding
   - Output: [batch_size, seq_len, d_model]

3. Decoder:
   - Input: target embeddings, encoder output
   - Process: Multiple decoder layers
   - Output: [batch_size, seq_len, d_model]

4. Final Linear Layer:
   - Input: decoder output [batch_size, seq_len, d_model]
   - Output: logits [batch_size, seq_len, num_labels]

5. Loss Computation:
   - Input: logits [batch_size, seq_len, num_labels], targets [batch_size, seq_len]
   - Process: Cross-entropy loss per position
   - Output: scalar loss
```

#### Backward Pass
```python
Input: loss, logits, targets
Flow:
1. Gradient Computation:
   - grad = compute_grad(logits, targets)

2. Final Linear Layer:
   - grad_final = grad @ final_linear.T
   - grads['final_linear'] = last_decoder_output.T @ grad

3. Decoder:
   - Backprop through decoder layers
   - Output: decoder gradients

4. Encoder:
   - Backprop through encoder layers
   - Update parameters
```

### 3. Generation Task

#### Forward Pass
```python
Input: src_batch [batch_size, seq_len], tgt_batch [batch_size, seq_len-1], masks
Flow:
1. Source Processing:
   - Source embeddings + positional encoding
   - Encoder processing
   - Output: [batch_size, seq_len, d_model]

2. Target Processing:
   - Target embeddings + positional encoding
   - Output: [batch_size, seq_len-1, d_model]

3. Decoder:
   - Input: target embeddings, encoder output
   - Process: Multiple decoder layers
   - Output: [batch_size, seq_len-1, d_model]

4. Final Linear Layer:
   - Input: decoder output [batch_size, seq_len-1, d_model]
   - Output: logits [batch_size, seq_len-1, vocab_size]

5. Loss Computation:
   - Input: logits [batch_size, seq_len-1, vocab_size], targets [batch_size, seq_len-1]
   - Process: Cross-entropy loss
   - Output: scalar loss
```

#### Backward Pass
```python
Input: loss, logits, targets
Flow:
1. Gradient Computation:
   - grad = compute_grad(logits, targets)

2. Final Linear Layer:
   - grad_final = grad @ final_linear.T
   - grads['final_linear'] = last_decoder_output.T @ grad

3. Decoder:
   - Backprop through decoder layers
   - Output: decoder gradients

4. Encoder:
   - Backprop through encoder layers
   - Update parameters
```

### 4. Question Answering Task

#### Forward Pass
```python
Input: src_batch [batch_size, seq_len], src_mask
Flow:
1. Source Processing:
   - Source embeddings + positional encoding
   - Encoder processing
   - Output: [batch_size, seq_len, d_model]

2. Pooling:
   - Input: encoder output [batch_size, seq_len, d_model]
   - Process: Mean pooling over sequence dimension
   - Output: [batch_size, d_model]

3. Final Linear Layer:
   - Input: pooled [batch_size, d_model]
   - Output: logits [batch_size, 2]  # start and end positions

4. Loss Computation:
   - Input: logits [batch_size, 2], targets [batch_size, 2]
   - Process: Cross-entropy loss
   - Output: scalar loss
```

#### Backward Pass
```python
Input: loss, logits, targets
Flow:
1. Gradient Computation:
   - grad = compute_grad(logits, targets)

2. Final Linear Layer:
   - grad_final = grad @ final_linear.T
   - grads['final_linear'] = last_pooled.T @ grad

3. Encoder:
   - encoder_grad = repeat(grad_final) / seq_len
   - Backprop through encoder layers

4. Parameter Updates:
   - Update using Adam optimizer
```

## Training Loop
```python
For each epoch:
    For each batch:
        1. Forward Pass
           - Compute predictions
           - Compute loss
        
        2. Backward Pass
           - Compute gradients
           - Update parameters
        
        3. Validation
           - Compute validation loss
           - Save checkpoint if improved
```

## Key Differences Between Tasks

1. **Classification**:
   - Uses only encoder
   - Pools encoder output
   - Single prediction per input

2. **Sequence Labeling**:
   - Uses both encoder and decoder
   - Predicts label for each position
   - No pooling

3. **Generation**:
   - Uses both encoder and decoder
   - Teacher forcing during training
   - Autoregressive during inference

4. **Question Answering**:
   - Uses only encoder
   - Predicts start and end positions
   - No decoder needed 