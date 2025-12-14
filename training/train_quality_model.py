#!/usr/bin/env python3
"""
Train a quality prediction model for answer assessment.
Supports both simple (logistic regression) and neural network models.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from feature_engineering import load_ratings, extract_features_batch, normalize_features


class QualityMLP(nn.Module):
    """Multi-layer perceptron for quality prediction."""

    def __init__(self, input_size: int, hidden_sizes: list = [32, 16]):
        super(QualityMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Output 0-1

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def train_linear_model(X_train, y_train, X_test, y_test):
    """Train a simple linear regression model."""
    print("Training Ridge Regression model...")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Clip predictions to [0, 1]
    y_pred_train = np.clip(y_pred_train, 0, 1)
    y_pred_test = np.clip(y_pred_test, 0, 1)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Train MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    return model, y_pred_test


def train_neural_network(X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    """Train a neural network model."""
    print("Training Neural Network model...")

    input_size = X_train.shape[1]
    model = QualityMLP(input_size)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t).numpy()
        y_pred_test = model(X_test_t).numpy()

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nFinal Results:")
    print(f"Train MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig('training/training_curve.png')
    print("Training curve saved to training/training_curve.png")

    return model, y_pred_test, (train_losses, test_losses)


def plot_predictions(y_true, y_pred, output_path='training/predictions.png'):
    """Plot predicted vs actual ratings."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlabel('Actual Rating (normalized)')
    plt.ylabel('Predicted Rating (normalized)')
    plt.title('Predicted vs Actual Ratings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Predictions plot saved to {output_path}")


def save_model_metadata(feature_names, mean, std, output_path='training/model_metadata.json'):
    """Save feature names and normalization parameters."""
    metadata = {
        'feature_names': feature_names,
        'mean': mean.tolist(),
        'std': std.tolist(),
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train quality prediction model')
    parser.add_argument('ratings_file', help='Path to ratings JSONL file')
    parser.add_argument('--model', choices=['linear', 'nn'], default='nn', help='Model type')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (NN only)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (NN only)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion')

    args = parser.parse_args()

    # Load and process data
    print(f"Loading ratings from {args.ratings_file}")
    ratings = load_ratings(args.ratings_file)
    print(f"Loaded {len(ratings)} ratings")

    if len(ratings) < 10:
        print("Error: Need at least 10 ratings to train a model")
        return

    # Extract features
    X, y, feature_names = extract_features_batch(ratings)
    print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")

    # Normalize
    X_normalized, mean, std = normalize_features(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=args.test_size, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    if args.model == 'linear':
        model, y_pred = train_linear_model(X_train, y_train, X_test, y_test)
    else:
        model, y_pred, losses = train_neural_network(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs, lr=args.lr
        )

    # Plot results
    plot_predictions(y_test, y_pred)

    # Save metadata
    save_model_metadata(feature_names, mean, std)

    print("\nTraining complete!")
    print("Next steps:")
    print("  1. Review training/predictions.png to check model performance")
    print("  2. Run export_onnx.py to convert the model for Go inference")


if __name__ == "__main__":
    main()
