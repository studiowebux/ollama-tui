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
