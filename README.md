# Ollama UI

Extensible terminal-based chatbot for Ollama with real-time streaming, chat persistence, and model selection.

## Features

- Model selection from available Ollama models
- Configurable endpoint (host:port)
- Local chat persistence with history
- Real-time streaming responses
- Terminal UI with multiple views (chat, chat list, settings)
- Keyboard shortcuts for navigation

## Requirements

- Go 1.23+
- Running Ollama instance

## Installation

```bash
go mod download
go build -o ollama-ui
```

## Usage

```bash
./ollama-ui
```

## Keyboard Shortcuts

### Chat View
- `Enter`: Send message
- `Ctrl+N`: New chat
- `Ctrl+L`: View chat list
- `Ctrl+S`: Settings
- `Ctrl+C`: Quit

### Chat List View
- `↑/↓` or `k/j`: Navigate chats
- `Enter`: Open selected chat
- `n`: New chat
- `d`: Delete selected chat
- `s`: Settings
- `q`: Quit

### Settings View
- `Tab`: Switch between endpoint and model
- `↑/↓` or `k/j`: Select model (when focused)
- `Enter`: Save and return
- `Esc`: Cancel and return

## Configuration

Configuration stored in `~/.ollama-ui/config.json`

Default endpoint: `http://localhost:11434`

## Chat Storage

Chats saved in `~/.ollama-ui/chats/`

## Machine Learning Quality Prediction

The refinement system can optionally use a trained neural network for quality prediction instead of heuristics.

### Using ML Model

1. Collect rating data using the TUI (press 'r' to rate answers)
2. Export ratings: `./ollamatui export-ratings -o ratings.jsonl`
3. Train model: `cd training && python train_quality_model.py ratings.jsonl --model nn`
4. Export to ONNX: `python export_onnx.py --model quality_model.pth`
5. Copy `quality_model.onnx` and `model_metadata.json` to project root

### Requirements

macOS with ONNX Runtime installed:
```bash
brew install onnxruntime
```

For other platforms, update `onnxruntime.SetSharedLibraryPath()` in ml_scorer.go

### Fallback Behavior

If ONNX model not found, system automatically falls back to heuristic quality scorer. No configuration needed.

See `training/README.md` for detailed ML pipeline documentation.
