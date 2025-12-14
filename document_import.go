package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DocumentType represents the type of document being imported
type DocumentType string

const (
	DocTypeMarkdown   DocumentType = "markdown"
	DocTypeGo         DocumentType = "go"
	DocTypeTypeScript DocumentType = "typescript"
	DocTypeJavaScript DocumentType = "javascript"
	DocTypePython     DocumentType = "python"
	DocTypeRust       DocumentType = "rust"
	DocTypeOther      DocumentType = "other"
)

// ImportedDocument represents a document imported into the KB
type ImportedDocument struct {
	ID           string       `json:"id"`
	FilePath     string       `json:"file_path"`
	RelativePath string       `json:"relative_path"`
	Type         DocumentType `json:"type"`
	Content      string       `json:"content"`
	Hash         string       `json:"hash"`
	ImportedAt   time.Time    `json:"imported_at"`
	LastModified time.Time    `json:"last_modified"`
}

// CodeSnippet represents a classified code segment
type CodeSnippet struct {
	Language    string `json:"language"`
	Code        string `json:"code"`
	Summary     string `json:"summary"`      // One-liner summary
	Context     string `json:"context"`      // Surrounding context (function name, class, etc)
	FilePath    string `json:"file_path"`
	StartLine   int    `json:"start_line"`
	EndLine     int    `json:"end_line"`
	SnippetType string `json:"snippet_type"` // function, class, method, snippet
}

// DocumentImporter handles importing and processing documents
type DocumentImporter struct {
	client   *OllamaClient
	vectorDB *VectorDB
	basePath string
}

func NewDocumentImporter(client *OllamaClient, vectorDB *VectorDB, basePath string) *DocumentImporter {
	return &DocumentImporter{
		client:   client,
		vectorDB: vectorDB,
		basePath: basePath,
	}
}

// SupportedExtensions returns file extensions to scan
func (di *DocumentImporter) SupportedExtensions() map[string]DocumentType {
	return map[string]DocumentType{
		".md":   DocTypeMarkdown,
		".go":   DocTypeGo,
		".ts":   DocTypeTypeScript,
		".tsx":  DocTypeTypeScript,
		".js":   DocTypeJavaScript,
		".jsx":  DocTypeJavaScript,
		".py":   DocTypePython,
		".rs":   DocTypeRust,
	}
}

// ScanDirectory recursively scans for supported files
func (di *DocumentImporter) ScanDirectory(dirPath string) ([]string, error) {
	var files []string
	supportedExts := di.SupportedExtensions()

	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			// Skip common directories to ignore
			name := info.Name()
			if name == "node_modules" || name == ".git" || name == "vendor" ||
			   name == "dist" || name == "build" || name == ".next" {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(path))
		if _, ok := supportedExts[ext]; ok {
			files = append(files, path)
		}

		return nil
	})

	return files, err
}

// ImportDocument imports a single document and vectorizes it using all strategies
func (di *DocumentImporter) ImportDocument(filePath, model, embedModel string, progressChan chan<- string) error {
	return di.ImportDocumentWithStrategy(filePath, model, embedModel, "all", false, progressChan)
}

// processMarkdown handles markdown documents
func (di *DocumentImporter) processMarkdown(doc ImportedDocument, model, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- fmt.Sprintf("Processing markdown: %s", doc.RelativePath)
	}

	// Split by headings for better chunking
	sections := di.splitMarkdownSections(doc.Content)

	for _, section := range sections {
		if strings.TrimSpace(section.Content) == "" {
			continue
		}

		// Generate summary for this section
		summary, err := di.generateMarkdownSummary(model, section.Heading, section.Content)
		if err != nil {
			summary = section.Heading
		}

		// Create embedding
		embedding, err := di.client.GenerateEmbedding(embedModel, section.Content)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     section.Content,
			ContentType: ContentTypeFact,
			Strategy:    "document_section",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:     section.Content,
				SearchKeywords:   []string{"markdown", "documentation", section.Heading},
				SourceDocument:   doc.RelativePath,
				DocumentType:     string(doc.Type),
				DocumentHash:     doc.Hash,
				Timestamp:        doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{summary}
		chunk.CanonicalAnswer = section.Content

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processCode handles code files with classification
func (di *DocumentImporter) processCode(doc ImportedDocument, model, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- fmt.Sprintf("Processing code: %s", doc.RelativePath)
	}

	// Extract code snippets with classification
	snippets, err := di.classifyCode(model, doc)
	if err != nil {
		return err
	}

	for _, snippet := range snippets {
		if snippet.Summary == "" {
			continue
		}

		// Generate embedding for the summary (what user will search for)
		embedding, err := di.client.GenerateEmbedding(embedModel, snippet.Summary)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     snippet.Summary, // Summary is what gets searched
			ContentType: ContentTypeCode,
			Strategy:    "code_snippet",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:     snippet.Code, // Full code stored here
				SearchKeywords:   []string{snippet.Language, snippet.SnippetType, snippet.Context},
				SourceDocument:   doc.RelativePath,
				DocumentType:     string(doc.Type),
				DocumentHash:     doc.Hash,
				CodeLanguage:     snippet.Language,
				CodeContext:      snippet.Context,
				Timestamp:        doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{snippet.Summary}
		chunk.CanonicalAnswer = snippet.Code

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processGeneric handles other file types
func (di *DocumentImporter) processGeneric(doc ImportedDocument, model, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- fmt.Sprintf("Processing file: %s", doc.RelativePath)
	}

	embedding, err := di.client.GenerateEmbedding(embedModel, doc.Content)
	if err != nil {
		return err
	}

	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     doc.Content,
		ContentType: ContentTypeFact,
		Strategy:    "document_full",
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:   doc.Content,
			SourceDocument: doc.RelativePath,
			DocumentType:   string(doc.Type),
			Timestamp:      doc.ImportedAt,
		},
	}

	return di.vectorDB.AddChunk(chunk)
}

// MarkdownSection represents a section of a markdown document
type MarkdownSection struct {
	Heading string
	Content string
	Level   int
}

// splitMarkdownSections splits markdown by headings
func (di *DocumentImporter) splitMarkdownSections(content string) []MarkdownSection {
	var sections []MarkdownSection
	lines := strings.Split(content, "\n")

	var currentHeading string
	var currentContent strings.Builder
	var currentLevel int

	for _, line := range lines {
		// Check for markdown heading
		if strings.HasPrefix(line, "#") {
			// Save previous section
			if currentContent.Len() > 0 {
				sections = append(sections, MarkdownSection{
					Heading: currentHeading,
					Content: currentContent.String(),
					Level:   currentLevel,
				})
				currentContent.Reset()
			}

			// Parse new heading
			level := 0
			for _, ch := range line {
				if ch == '#' {
					level++
				} else {
					break
				}
			}
			currentLevel = level
			currentHeading = strings.TrimSpace(strings.TrimLeft(line, "#"))
		} else {
			currentContent.WriteString(line)
			currentContent.WriteString("\n")
		}
	}

	// Save last section
	if currentContent.Len() > 0 {
		sections = append(sections, MarkdownSection{
			Heading: currentHeading,
			Content: currentContent.String(),
			Level:   currentLevel,
		})
	}

	return sections
}

// generateMarkdownSummary creates a summary for a markdown section
func (di *DocumentImporter) generateMarkdownSummary(model, heading, content string) (string, error) {
	prompt := fmt.Sprintf(`Generate a concise question that this documentation section answers.

Heading: %s
Content: %s

Return ONLY the question (one line, no quotes):`, heading, content[:min(500, len(content))])

	messages := []ChatMessage{
		{Role: "user", Content: prompt},
	}
	response, err := di.client.Chat(model, messages)
	if err != nil {
		return heading, err
	}

	return strings.TrimSpace(response), nil
}

// classifyCode extracts and classifies code snippets
func (di *DocumentImporter) classifyCode(model string, doc ImportedDocument) ([]CodeSnippet, error) {
	language := string(doc.Type)

	prompt := fmt.Sprintf(`Analyze this %s code and extract meaningful code snippets with one-liner summaries.

For each function, method, class, or significant code block, provide:
1. The exact code
2. A one-liner summary (what it does, not how)
3. Context (function/class name)
4. Type (function/class/method/snippet)

File: %s
Code:
%s

Return ONLY a JSON array (no markdown, no explanation):
[
  {
    "code": "the exact code snippet",
    "summary": "one-line description of what it does",
    "context": "function or class name",
    "snippet_type": "function|class|method|snippet"
  }
]`, language, doc.RelativePath, doc.Content)

	messages := []ChatMessage{
		{Role: "user", Content: prompt},
	}
	response, err := di.client.Chat(model, messages)
	if err != nil {
		return nil, err
	}

	// Extract JSON
	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in response")
	}

	// Parse response
	var results []struct {
		Code        string `json:"code"`
		Summary     string `json:"summary"`
		Context     string `json:"context"`
		SnippetType string `json:"snippet_type"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &results); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	snippets := make([]CodeSnippet, 0, len(results))
	for _, r := range results {
		snippets = append(snippets, CodeSnippet{
			Language:    language,
			Code:        r.Code,
			Summary:     r.Summary,
			Context:     r.Context,
			FilePath:    doc.RelativePath,
			SnippetType: r.SnippetType,
		})
	}

	return snippets, nil
}
