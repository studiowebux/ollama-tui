package main

import (
	"fmt"
	"strings"
)

// ContextCompressor reduces verbose context to key facts
type ContextCompressor struct {
	client *OllamaClient
	model  string
}

// NewContextCompressor creates a context compressor
func NewContextCompressor(client *OllamaClient, model string) *ContextCompressor {
	return &ContextCompressor{
		client: client,
		model:  model,
	}
}

// CompressContext extracts key facts relevant to the query
func (c *ContextCompressor) CompressContext(query string, chunks []SearchResult, maxChunks int) (string, error) {
	if len(chunks) == 0 {
		return "", nil
	}

	// Limit to top N chunks
	if len(chunks) > maxChunks {
		chunks = chunks[:maxChunks]
	}

	// For single chunk, return as-is if short enough
	if len(chunks) == 1 && len(chunks[0].Chunk.Content) < 500 {
		return chunks[0].Chunk.Content, nil
	}

	// Build context from chunks
	var contextParts []string
	for i, chunk := range chunks {
		contextParts = append(contextParts, fmt.Sprintf("Chunk %d:\n%s", i+1, chunk.Chunk.Content))
	}

	fullContext := strings.Join(contextParts, "\n\n")

	// Ask LLM to extract only relevant facts
	prompt := fmt.Sprintf(`You are extracting key facts to answer a question.

Question: %s

Context chunks:
%s

Extract ONLY the facts directly relevant to answering the question. Be extremely concise.
Format: One fact per line, no extra explanation.
Maximum 5 facts total.

Relevant facts:`, query, fullContext)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	compressed, err := c.client.Chat(c.model, messages)
	if err != nil {
		// Fallback to first chunk if compression fails
		return chunks[0].Chunk.Content, nil
	}

	// If compression resulted in something too long, truncate to first chunk
	if len(compressed) > len(fullContext)/2 {
		return chunks[0].Chunk.Content, nil
	}

	return strings.TrimSpace(compressed), nil
}

// RerankChunks scores chunks by relevance to query
func (c *ContextCompressor) RerankChunks(query string, chunks []SearchResult) ([]SearchResult, error) {
	if len(chunks) <= 3 {
		return chunks, nil // Already small enough
	}

	// Use LLM to score relevance of each chunk
	// For now, use simple heuristic: prefer chunks with query terms
	queryWords := strings.Fields(strings.ToLower(query))

	for i := range chunks {
		content := strings.ToLower(chunks[i].Chunk.Content)

		// Boost score if chunk contains query keywords
		matches := 0
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(content, word) {
				matches++
			}
		}

		// Boost by keyword coverage
		if len(queryWords) > 0 {
			coverage := float64(matches) / float64(len(queryWords))
			chunks[i].Similarity += coverage * 0.2 // Boost up to 20%
		}
	}

	// Sort by adjusted similarity
	for i := 0; i < len(chunks)-1; i++ {
		for j := i + 1; j < len(chunks); j++ {
			if chunks[j].Similarity > chunks[i].Similarity {
				chunks[i], chunks[j] = chunks[j], chunks[i]
			}
		}
	}

	return chunks, nil
}
