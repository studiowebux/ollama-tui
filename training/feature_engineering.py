#!/usr/bin/env python3
"""
Feature engineering for answer quality prediction.
Extracts features from rating data exported from ollamatui.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import re


def load_ratings(jsonl_path: str) -> List[Dict]:
    """Load ratings from JSONL file."""
    ratings = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                ratings.append(json.loads(line))
    return ratings


def extract_significant_words(text: str, stopwords: set) -> List[str]:
    """Extract significant words from text (excluding stopwords)."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if len(w) > 2 and w not in stopwords]


def calculate_query_coverage(query: str, answer: str) -> float:
    """Calculate percentage of query terms in answer."""
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'how', 'what', 'when', 'where', 'who', 'why'
    }

    query_words = extract_significant_words(query, stopwords)
    answer_lower = answer.lower()

    if not query_words:
        return 0.0

    matched = sum(1 for word in query_words if word in answer_lower)
    return matched / len(query_words)


def calculate_answer_completeness(answer: str) -> float:
    """Calculate answer completeness score based on length and structure."""
    length = len(answer)

    # Length score
    if length < 50:
        length_score = 0.3
    elif length < 150:
        length_score = 0.6
    elif length < 500:
        length_score = 0.8
    else:
        length_score = 1.0

    # Structure bonus
    structure_bonus = 0.0
    if '\n\n' in answer:
        structure_bonus += 0.1
    if '```' in answer or '- ' in answer or '* ' in answer:
        structure_bonus += 0.1

    return min(1.0, length_score + structure_bonus)


def extract_features(rating: Dict) -> Dict[str, float]:
    """Extract all features from a single rating entry."""
    features = {}

    # Basic metadata features
    features['context_used'] = 1.0 if rating['context_used'] else 0.0
    features['context_chunks'] = float(rating['context_chunks'])
    features['vector_top_k'] = float(rating['vector_top_k'])
    features['vector_similarity'] = float(rating['vector_similarity'])

    # Text-based features
    query = rating['query']
    answer = rating['answer']

    features['query_length'] = float(len(query))
    features['answer_length'] = float(len(answer))
    features['answer_query_ratio'] = len(answer) / max(len(query), 1)

    features['query_coverage'] = calculate_query_coverage(query, answer)
    features['answer_completeness'] = calculate_answer_completeness(answer)

    # Word-level features
    query_words = len(query.split())
    answer_words = len(answer.split())

    features['query_word_count'] = float(query_words)
    features['answer_word_count'] = float(answer_words)
    features['words_per_chunk'] = answer_words / max(rating['context_chunks'], 1)

    # Structural features
    features['has_paragraphs'] = 1.0 if '\n\n' in answer else 0.0
    features['has_code_blocks'] = 1.0 if '```' in answer else 0.0
    features['has_lists'] = 1.0 if ('- ' in answer or '* ' in answer) else 0.0

    # Target variable (normalize rating to 0-1)
    features['rating_score'] = (rating['rating'] - 1) / 4.0  # 1-5 -> 0-1

    return features


def extract_features_batch(ratings: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract features from all ratings.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature names
    """
    all_features = []

    for rating in ratings:
        features = extract_features(rating)
        all_features.append(features)

    if not all_features:
        raise ValueError("No ratings to process")

    # Get feature names (exclude target)
    feature_names = [k for k in all_features[0].keys() if k != 'rating_score']

    # Build feature matrix
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array([f['rating_score'] for f in all_features])

    return X, y, feature_names


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.

    Returns:
        X_normalized: Normalized features
        mean: Feature means
        std: Feature standard deviations
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <ratings.jsonl>")
        sys.exit(1)

    ratings = load_ratings(sys.argv[1])
    print(f"Loaded {len(ratings)} ratings")

    X, y, feature_names = extract_features_batch(ratings)
    print(f"Extracted {X.shape[1]} features")
    print(f"Feature names: {feature_names}")

    print("\nFeature statistics:")
    for i, name in enumerate(feature_names):
        print(f"  {name:25s} mean={X[:, i].mean():.3f} std={X[:, i].std():.3f}")

    print(f"\nTarget statistics:")
    print(f"  Rating mean: {y.mean():.3f}")
    print(f"  Rating std: {y.std():.3f}")
    print(f"  Rating range: [{y.min():.3f}, {y.max():.3f}]")
