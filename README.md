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

### Setup

1. Install ONNX Runtime:
   - macOS: `brew install onnxruntime`
   - Linux: Download from https://github.com/microsoft/onnxruntime/releases
   - Platform-specific library path will be auto-detected

2. Collect rating data using the TUI (press 'r' to rate answers, need 100+ samples)

3. Export ratings:
   ```bash
   ./ollamatui export-ratings -o ratings.jsonl
   ```

4. Train model:
   ```bash
   cd training
   python train_quality_model.py ratings.jsonl --model nn --epochs 100
   python export_onnx.py --model quality_model.pth
   ```

5. Configure in `~/.ollamatui/config.json`:
   ```json
   {
     "ml_enable_scoring": true,
     "ml_model_path": "/absolute/path/to/quality_model.onnx",
     "ml_metadata_path": "/absolute/path/to/model_metadata.json",
     "ml_onnx_lib_path": ""
   }
   ```

   Leave `ml_onnx_lib_path` empty for platform defaults.

### Behavior

- `ml_enable_scoring: false` (default) → Always uses heuristic scoring
- `ml_enable_scoring: true` + paths configured → Uses ML if model loads successfully
- If ML model fails to load → Warning logged, falls back to heuristic
- If ML inference fails → Falls back to heuristic for that query

No silent failures. All errors are logged to stderr.

See `training/README.md` for detailed ML pipeline documentation.
