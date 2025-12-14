# ML Training Pipeline for Quality Prediction

This directory contains the machine learning training pipeline for predicting answer quality scores.

## Overview

The pipeline trains a neural network or linear model to predict answer quality (0-1 score) based on features extracted from query-answer pairs and vector search metadata.

## Features Used

### Metadata Features (4)
- `context_used` - Whether vector context was used (0/1)
- `context_chunks` - Number of chunks retrieved
- `vector_top_k` - TopK setting used
- `vector_similarity` - Similarity threshold used

### Text-Based Features (6)
- `query_length` - Length of query in characters
- `answer_length` - Length of answer in characters
- `answer_query_ratio` - Ratio of answer to query length
- `query_coverage` - Percentage of query terms in answer
- `answer_completeness` - Answer quality based on length and structure
- `words_per_chunk` - Average words per retrieved chunk

### Word-Level Features (2)
- `query_word_count` - Number of words in query
- `answer_word_count` - Number of words in answer

### Structural Features (3)
- `has_paragraphs` - Answer has paragraph breaks (0/1)
- `has_code_blocks` - Answer has code blocks (0/1)
- `has_lists` - Answer has lists (0/1)

**Total: 15 features**

## Setup

Install dependencies:

```bash
cd training
pip install -r requirements.txt
```

## Usage

### Step 1: Export Ratings

First, export rated conversations from ollamatui:

```bash
./ollamatui export-ratings --project myproject -o ratings.jsonl
```

You need at least 10-20 rated conversations, but 100+ is recommended for good performance.

### Step 2: Inspect Features

Check the extracted features:

```bash
python feature_engineering.py ratings.jsonl
```

This will show feature statistics and help validate your data.

### Step 3: Train Model

Train a neural network (recommended):

```bash
python train_quality_model.py ratings.jsonl --model nn --epochs 100 --lr 0.001
```

Or train a simple linear model:

```bash
python train_quality_model.py ratings.jsonl --model linear
```

**Output files:**
- `training_curve.png` - Training progress visualization
- `predictions.png` - Predicted vs actual ratings scatter plot
- `model_metadata.json` - Feature names and normalization parameters
- `quality_model.pth` (or `.pkl` for linear) - Trained model weights

### Step 4: Export to ONNX

Convert the model for Go inference:

```bash
python export_onnx.py --model quality_model.pth --type pytorch --output quality_model.onnx
```

**Output files:**
- `quality_model.onnx` - Model in ONNX format for Go
- Verification output showing PyTorch vs ONNX difference

### Step 5: Deploy in Go

The ONNX model and metadata are ready for integration in the Go codebase (Feature 4 Part 3).

## Model Architecture

### Neural Network (Default)
- Input layer: 15 features
- Hidden layer 1: 32 neurons + ReLU + Dropout(0.2)
- Hidden layer 2: 16 neurons + ReLU + Dropout(0.2)
- Output layer: 1 neuron + Sigmoid (0-1 output)
- Loss: MSE
- Optimizer: Adam

### Linear Model (Alternative)
- Ridge Regression with alpha=1.0
- Faster to train, interpretable
- Good baseline for comparison

## Performance Metrics

The training script reports:
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: How well the model explains variance
- **Training curve**: Loss over epochs (NN only)
- **Predictions plot**: Visual check of model accuracy

**Good performance:**
- MAE < 0.15 (less than 1 star error on 5-star scale)
- R² > 0.5 (explains >50% of variance)
- Training and test metrics close (no overfitting)

## Troubleshooting

**Not enough data:**
- Need minimum 10 ratings, 100+ recommended
- Collect more ratings by using the TUI and pressing 'r'

**Poor performance:**
- Check predictions.png for patterns
- Try more training epochs: `--epochs 200`
- Collect more diverse ratings (different queries, contexts)
- Try linear model for small datasets

**ONNX export fails:**
- Install onnxruntime: `pip install onnxruntime`
- Check PyTorch version compatibility

## Data Collection Tips

For best ML performance:
1. Rate a diverse set of answers (good and bad)
2. Include different query types (factual, code, explanations)
3. Rate with and without vector context
4. Use different models to get variety
5. Aim for balanced ratings (not all 5 stars)

## Integration with Go

After exporting to ONNX:
1. Copy `quality_model.onnx` to Go project root
2. Copy `model_metadata.json` to Go project root
3. Implement Feature 4 Part 3 (ONNX inference in Go)
4. Replace heuristic scorer with ML scorer

See parent README for Go integration instructions.
