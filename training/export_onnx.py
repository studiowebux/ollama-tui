#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format for Go inference.
"""

import argparse
import json
import torch
import torch.onnx
import numpy as np

from train_quality_model import QualityMLP


def export_pytorch_to_onnx(model_path: str, metadata_path: str, output_path: str):
    """Export PyTorch model to ONNX format."""

    # Load metadata to get input size
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    input_size = len(metadata['feature_names'])
    print(f"Model input size: {input_size}")

    # Create model and load weights
    model = QualityMLP(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {output_path}")

    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")

    # Test inference
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)

    # Run test inference
    input_data = dummy_input.numpy()
    onnx_output = session.run(['output'], {'input': input_data})[0]
    pytorch_output = model(dummy_input).detach().numpy()

    # Compare outputs
    diff = np.abs(onnx_output - pytorch_output).max()
    print(f"Max difference between PyTorch and ONNX: {diff:.6f}")

    if diff < 1e-5:
        print("Export verification: PASSED")
    else:
        print("Warning: Outputs differ significantly")


def export_sklearn_to_onnx(model_path: str, metadata_path: str, output_path: str):
    """Export scikit-learn model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import pickle
    except ImportError:
        print("Error: skl2onnx not installed. Install with: pip install skl2onnx")
        return

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    input_size = len(metadata['feature_names'])

    # Define input type
    initial_type = [('float_input', FloatTensorType([None, input_size]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth or .pkl)')
    parser.add_argument('--metadata', default='training/model_metadata.json',
                        help='Path to model metadata JSON')
    parser.add_argument('--output', default='training/quality_model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--type', choices=['pytorch', 'sklearn'], default='pytorch',
                        help='Model type')

    args = parser.parse_args()

    if args.type == 'pytorch':
        export_pytorch_to_onnx(args.model, args.metadata, args.output)
    else:
        export_sklearn_to_onnx(args.model, args.metadata, args.output)

    print("\nNext steps:")
    print("  1. Copy training/quality_model.onnx to your Go project")
    print("  2. Copy training/model_metadata.json to your Go project")
    print("  3. Use the ONNX runtime in Go to load and run inference")


if __name__ == "__main__":
    main()
