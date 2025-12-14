package main

import (
	"fmt"
	"sort"
	"strings"
)

// RAGEngine handles retrieval-augmented generation
type RAGEngine struct {
	client     *OllamaClient
	vectorDB   *VectorDB
	config     *Config
	compressor *ContextCompressor
}

// NewRAGEngine creates a new RAG engine
func NewRAGEngine(client *OllamaClient, vectorDB *VectorDB, config *Config) *RAGEngine {
	compressor := NewContextCompressor(client, config.Model)
	return &RAGEngine{
		client:     client,
		vectorDB:   vectorDB,
		config:     config,
		compressor: compressor,
	}
}

// RAGResult contains the retrieved context and metadata
type RAGResult struct {
	Context        string
	Results        []SearchResult
	DebugInfo      string
	ContextUsed    bool
	QueriesUsed    []string
	ResultsCount   int
	ContextsUsed   int
}

// RetrieveContext searches vector DB for relevant context
func (r *RAGEngine) RetrieveContext(query string) (*RAGResult, error) {
	result := &RAGResult{
		QueriesUsed: make([]string, 0),
	}

	if !r.config.VectorEnabled {
		result.DebugInfo = "[Vector DB disabled]"
		return result, nil
	}

	// Enhance query if explicitly enabled
	var searchQueries []string
	searchQueries = append(searchQueries, query)
	result.QueriesUsed = append(result.QueriesUsed, query)

	if r.config.VectorEnhanceQuery {
		if enhancement, err := r.client.EnhanceQuery(r.config.Model, query); err == nil && enhancement != nil {
			// Add canonical form
			if enhancement.CanonicalForm != "" {
				searchQueries = append(searchQueries, enhancement.CanonicalForm)
				result.QueriesUsed = append(result.QueriesUsed, enhancement.CanonicalForm)
			}
			// Add enhanced queries (limit to 2 most relevant)
			for i, eq := range enhancement.EnhancedQueries {
				if i >= 2 {
					break
				}
				searchQueries = append(searchQueries, eq)
				result.QueriesUsed = append(result.QueriesUsed, eq)
			}
		}
	}

	// Search with all query variations and combine results
	allResults := make(map[string]SearchResult)

	for _, sq := range searchQueries {
		embedding, err := r.client.GenerateEmbedding(r.config.VectorModel, sq)
		if err != nil {
			continue
		}

		// Use hybrid search for better keyword matching
		results := r.vectorDB.SearchHybrid(embedding, sq, r.config.VectorTopK*2, r.config.VectorFuzzyThreshold)

		// Merge results, keeping highest similarity score for each chunk
		for _, searchResult := range results {
			if existing, ok := allResults[searchResult.Chunk.ID]; !ok || searchResult.Similarity > existing.Similarity {
				allResults[searchResult.Chunk.ID] = searchResult
			}
		}
	}

	// Convert map to slice and sort by similarity
	results := make([]SearchResult, 0, len(allResults))
	for _, searchResult := range allResults {
		results = append(results, searchResult)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// After sorting, keep only top results that meet similarity threshold
	// Filter by threshold first
	filtered := make([]SearchResult, 0)
	for _, result := range results {
		if result.Similarity >= r.config.VectorSimilarity {
			filtered = append(filtered, result)
		}
	}
	results = filtered

	// Limit to topK with sanity cap
	// Enforce max 20 chunks regardless of config (more than 20 overwhelms LLMs)
	maxChunks := r.config.VectorTopK
	cappedTopK := false
	if maxChunks > 20 {
		maxChunks = 20
		cappedTopK = true
	}
	if len(results) > maxChunks {
		results = results[:maxChunks]
	}

	// Optionally expand with related chunks
	if r.config.VectorIncludeRelated && len(results) > 0 {
		expanded := make(map[string]SearchResult)
		for _, searchResult := range results {
			expanded[searchResult.Chunk.ID] = searchResult

			// Add parent context
			if searchResult.Chunk.Metadata.ParentChunkID != "" {
				parent := r.vectorDB.GetChunkByID(searchResult.Chunk.Metadata.ParentChunkID)
				if parent != nil {
					expanded[parent.ID] = SearchResult{
						Chunk:      *parent,
						Similarity: searchResult.Similarity * 0.9,
					}
				}
			}

			// Add related chunks
			for _, relatedID := range searchResult.Chunk.Metadata.RelatedChunkIDs {
				related := r.vectorDB.GetChunkByID(relatedID)
				if related != nil {
					expanded[related.ID] = SearchResult{
						Chunk:      *related,
						Similarity: searchResult.Similarity * 0.85,
					}
				}
			}
		}

		// Convert back to slice and sort
		results = make([]SearchResult, 0, len(expanded))
		for _, searchResult := range expanded {
			results = append(results, searchResult)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})

		// Limit to topK after expansion
		if len(results) > r.config.VectorTopK {
			results = results[:r.config.VectorTopK]
		}
	}

	result.Results = results
	result.ResultsCount = len(results)

	var debugBuilder strings.Builder
	debugBuilder.WriteString(fmt.Sprintf("Query: %s\n", truncateString(query, 60)))
	debugBuilder.WriteString(fmt.Sprintf("Found %d results from vector DB\n", len(results)))
	if cappedTopK {
		debugBuilder.WriteString(fmt.Sprintf("Warning: vector_top_k=%d is too high, capped at 20 chunks\n", r.config.VectorTopK))
	}

	if len(results) == 0 {
		result.DebugInfo = debugBuilder.String() + "No results found."
		return result, nil
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("Relevant context from past conversations:\n\n")
	usedCount := 0

	for i, searchResult := range results {
		debugBuilder.WriteString(fmt.Sprintf("  %d. Similarity=%.4f (threshold=%.2f) ",
			i+1, searchResult.Similarity, r.config.VectorSimilarity))

		// Determine source and content based on chunk type
		var question, answer string
		if searchResult.Chunk.Metadata.SourceDocument != "" {
			// Document import chunk
			question = truncateString(searchResult.Chunk.Content, 60)
			answer = truncateString(searchResult.Chunk.CanonicalAnswer, 60)
			if searchResult.Chunk.Metadata.SourceDocument != "" {
				debugBuilder.WriteString(fmt.Sprintf("[%s] ", searchResult.Chunk.Metadata.SourceDocument))
			}
		} else {
			// Conversation chunk
			question = truncateString(searchResult.Chunk.Metadata.UserMessage, 60)
			answer = truncateString(searchResult.Chunk.Metadata.AssistantMessage, 60)
		}

		if searchResult.Similarity < r.config.VectorSimilarity {
			debugBuilder.WriteString("SKIPPED\n")
			debugBuilder.WriteString(fmt.Sprintf("     Q: %s\n", question))
			debugBuilder.WriteString(fmt.Sprintf("     A: %s\n", answer))
			continue
		}

		debugBuilder.WriteString("USED\n")
		debugBuilder.WriteString(fmt.Sprintf("     Q: %s\n", question))
		debugBuilder.WriteString(fmt.Sprintf("     A: %s\n", answer))
		usedCount++

		// Build context based on chunk type
		if searchResult.Chunk.Metadata.SourceDocument != "" {
			contextBuilder.WriteString(fmt.Sprintf("Q: %s\nA: %s\n\n",
				searchResult.Chunk.Content,
				searchResult.Chunk.CanonicalAnswer))
		} else {
			contextBuilder.WriteString(fmt.Sprintf("Q: %s\nA: %s\n\n",
				searchResult.Chunk.Metadata.UserMessage,
				searchResult.Chunk.Metadata.AssistantMessage))
		}
	}

	debugBuilder.WriteString(fmt.Sprintf("\nTotal contexts injected: %d\n", usedCount))
	result.DebugInfo = debugBuilder.String()
	result.ContextsUsed = usedCount

	if usedCount > 0 {
		result.ContextUsed = true
		rawContext := contextBuilder.String()

		// Optionally compress context to key facts
		if r.config.VectorCompressContext {
			// Only compress if we have many chunks or very long context
			if usedCount > 5 || len(rawContext) > 2000 {
				compressed, err := r.compressor.CompressContext(query, results, r.config.VectorTopK)
				if err == nil && compressed != "" {
					result.Context = "Key facts from knowledge base:\n\n" + compressed
					debugBuilder.WriteString(fmt.Sprintf("Context compressed: %d → %d chars\n", len(rawContext), len(compressed)))
					result.DebugInfo = debugBuilder.String()
				} else {
					result.Context = rawContext
				}
			} else {
				result.Context = rawContext
			}
		} else {
			result.Context = rawContext
		}
	}

	return result, nil
}
