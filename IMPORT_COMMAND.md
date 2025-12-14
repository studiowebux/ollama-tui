# Import Command Documentation

## Overview

The `ollamatui import` command allows you to import documents (markdown, code files) into the vector database for RAG (Retrieval Augmented Generation) functionality.

## Installation

```bash
# Build the binary
go build -o ollamatui .

# Install completion (optional)
# For zsh
mkdir -p ~/.zsh/completions
ollamatui completion zsh > ~/.zsh/completions/_ollamatui
# Add to ~/.zshrc: fpath=(~/.zsh/completions $fpath)

# For bash
ollamatui completion bash > /etc/bash_completion.d/ollamatui
```

## Basic Usage

```bash
# Import a single file
ollamatui import /path/to/document.md

# Import an entire directory
ollamatui import /path/to/docs

# Import with verbose output
ollamatui import /path/to/docs --verbose

# Import to specific project
ollamatui import /path/to/docs --project myproject

# Import with specific models
ollamatui import /path/to/docs \
  --chat-model phi4:latest \
  --embed-model nomic-embed-text

# Force re-import (skip hash check)
ollamatui import /path/to/docs --force
```

## Flags

- `--project <id>`: Target project (default: current project from config)
- `--chat-model <model>`: Model for generating summaries (default: from config)
- `--embed-model <model>`: Model for embeddings (default: from config)
- `--force`: Re-import already imported files (skip hash check)
- `--verbose`: Show detailed progress during import

## Auto-completion

The import command supports intelligent auto-completion:

### Project Completion
```bash
ollamatui import ./docs --project <TAB>
# Shows: default, myproject, etc.
```

### Chat Model Completion
```bash
ollamatui import ./docs --chat-model <TAB>
# Shows: phi4:latest, qwen2.5:3b, etc. (excludes embed-only models)
```

### Embed Model Completion
```bash
ollamatui import ./docs --embed-model <TAB>
# Shows: nomic-embed-text, mxbai-embed-large, all-minilm, etc.
```

Auto-completion queries Ollama API to fetch available models and filters them appropriately.

## Supported File Types

The importer automatically detects and processes:

- **Markdown** (.md): Split by headings, generate summaries
- **Go** (.go): Extract functions, methods, classes
- **TypeScript** (.ts, .tsx): Extract code snippets
- **JavaScript** (.js, .jsx): Extract code snippets
- **Python** (.py): Extract code snippets
- **Rust** (.rs): Extract code snippets

## Import Process

1. **Scan**: Recursively scan directory for supported files
2. **Hash Check**: Skip already imported files (unless --force)
3. **Process**:
   - Markdown: Split into sections, generate canonical questions
   - Code: Extract snippets with summaries
4. **Embed**: Generate embeddings using specified model
5. **Store**: Save chunks to vector database with metadata

## Output Example

```
╔════════════════════════════════════════════════════╗
║           Document Import to VectorDB              ║
╚════════════════════════════════════════════════════╝

Project: Default Project
Chat Model: phi4:latest
Embed Model: nomic-embed-text
Path: ./docs

Scanning directory...
Found 15 files to process

[1/15] Processing: README.md
  ✓ Imported
[2/15] Processing: API.md
  ✓ Imported
[3/15] Processing: old-doc.md
  ⊗ Skipped (already imported)
...

╔════════════════════════════════════════════════════╗
║                Import Summary                      ║
╚════════════════════════════════════════════════════╝

Files Scanned:         15
Successfully Imported: 12
Already Imported:      2
Failed:                1

Total Chunks Created:  47
Storage Path:          ~/.ollamatui/projects/default/vectors
Total Chunks in DB:    153
```

## Environment Variables

- `OLLAMA_ENDPOINT`: Override Ollama server endpoint (default: http://localhost:11434)

## Error Handling

- **Path not found**: Displays error and exits
- **Ollama connection failed**: Shows endpoint, suggests checking Ollama is running
- **Model not found**: Lists available models, suggests `ollama pull`
- **File errors**: Logs error, continues with next file
- **Empty files**: Skips with warning

## Use Cases

### Import Documentation
```bash
ollamatui import ./docs --chat-model qwen2.5:3b --embed-model nomic-embed-text
```

### Import Codebase
```bash
ollamatui import ./src --verbose
```

### Update Changed Files
```bash
ollamatui import ./docs --force
```

## Notes

- Hash-based deduplication prevents re-importing unchanged files
- Content store deduplicates identical content across chunks
- Imported documents are searchable immediately in the TUI
- Use `--verbose` to see which sections/snippets are extracted
