package main

import (
	"strings"
)

// QualityScore represents the quality assessment of an answer
type QualityScore struct {
	OverallScore       float64            // 0-1 overall quality score
	SemanticRelevance  float64            // Average similarity of retrieved chunks
	QueryCoverage      float64            // Percentage of query terms in answer
	AnswerCompleteness float64            // Answer length/depth score
	ContextUsage       float64            // How much context was used in answer
	Details            map[string]float64 // Individual component scores
}

// CalculateQualityScore computes a heuristic quality score for an answer
func CalculateQualityScore(query, answer string, ragResult *RAGResult) *QualityScore {
	score := &QualityScore{
		Details: make(map[string]float64),
	}

	// 1. Semantic Relevance (30% weight) - Average similarity of retrieved chunks
	if ragResult != nil && len(ragResult.Results) > 0 {
		totalSim := 0.0
		for _, result := range ragResult.Results {
			totalSim += result.Similarity
		}
		score.SemanticRelevance = totalSim / float64(len(ragResult.Results))
	}
	score.Details["semantic_relevance"] = score.SemanticRelevance

	// 2. Query Coverage (30% weight) - Percentage of query terms appearing in answer
	queryWords := extractSignificantWords(query)
	answerLower := strings.ToLower(answer)
	matchedWords := 0
	for _, word := range queryWords {
		if strings.Contains(answerLower, strings.ToLower(word)) {
			matchedWords++
		}
	}
	if len(queryWords) > 0 {
		score.QueryCoverage = float64(matchedWords) / float64(len(queryWords))
	}
	score.Details["query_coverage"] = score.QueryCoverage

	// 3. Answer Completeness (20% weight) - Based on answer length and structure
	score.AnswerCompleteness = calculateCompleteness(answer)
	score.Details["answer_completeness"] = score.AnswerCompleteness

	// 4. Context Usage (20% weight) - How much retrieved context appears in answer
	if ragResult != nil && ragResult.ContextUsed {
		score.ContextUsage = calculateContextUsage(answer, ragResult)
	}
	score.Details["context_usage"] = score.ContextUsage

	// Calculate weighted overall score
	score.OverallScore = (score.SemanticRelevance * 0.30) +
		(score.QueryCoverage * 0.30) +
		(score.AnswerCompleteness * 0.20) +
		(score.ContextUsage * 0.20)

	return score
}

// extractSignificantWords extracts meaningful words from text (excluding common stopwords)
func extractSignificantWords(text string) []string {
	stopwords := map[string]bool{
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
		"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
		"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
		"that": true, "the": true, "to": true, "was": true, "will": true, "with": true,
		"how": true, "what": true, "when": true, "where": true, "who": true, "why": true,
	}

	words := strings.Fields(strings.ToLower(text))
	significant := make([]string, 0)

	for _, word := range words {
		// Remove punctuation
		word = strings.Trim(word, ".,!?;:\"'")
		if len(word) > 2 && !stopwords[word] {
			significant = append(significant, word)
		}
	}

	return significant
}

// calculateCompleteness scores answer based on length and structure
func calculateCompleteness(answer string) float64 {
	// Score based on answer length
	answerLength := len(answer)

	var lengthScore float64
	if answerLength < 50 {
		lengthScore = 0.3 // Very short answer
	} else if answerLength < 150 {
		lengthScore = 0.6 // Short answer
	} else if answerLength < 500 {
		lengthScore = 0.8 // Medium answer
	} else {
		lengthScore = 1.0 // Detailed answer
	}

	// Bonus for structure (lists, paragraphs, code blocks)
	structureBonus := 0.0
	if strings.Contains(answer, "\n\n") {
		structureBonus += 0.1 // Has paragraphs
	}
	if strings.Contains(answer, "```") || strings.Contains(answer, "- ") || strings.Contains(answer, "* ") {
		structureBonus += 0.1 // Has code blocks or lists
	}

	score := lengthScore + structureBonus
	if score > 1.0 {
		score = 1.0
	}

	return score
}

// calculateContextUsage measures how much of the retrieved context appears in the answer
func calculateContextUsage(answer string, ragResult *RAGResult) float64 {
	if ragResult == nil || !ragResult.ContextUsed || len(ragResult.Results) == 0 {
		return 0.0
	}

	answerLower := strings.ToLower(answer)
	totalChunks := len(ragResult.Results)
	chunksReferenced := 0

	// Check how many chunks have significant content overlap with the answer
	for _, result := range ragResult.Results {
		chunkContent := result.Chunk.Content
		if chunkContent == "" {
			chunkContent = result.Chunk.CanonicalAnswer
		}

		// Extract significant words from chunk
		chunkWords := extractSignificantWords(chunkContent)
		matchedWords := 0

		for _, word := range chunkWords {
			if strings.Contains(answerLower, strings.ToLower(word)) {
				matchedWords++
			}
		}

		// If more than 30% of chunk's significant words appear in answer, count it as referenced
		if len(chunkWords) > 0 && float64(matchedWords)/float64(len(chunkWords)) > 0.3 {
			chunksReferenced++
		}
	}

	if totalChunks > 0 {
		return float64(chunksReferenced) / float64(totalChunks)
	}

	return 0.0
}

// ShouldRefine determines if an answer should be refined based on quality score
func ShouldRefine(score *QualityScore, threshold float64) bool {
	return score.OverallScore < threshold
}

// IdentifyWeaknesses returns a list of specific weaknesses in the answer
func IdentifyWeaknesses(score *QualityScore) []string {
	weaknesses := make([]string, 0)

	if score.SemanticRelevance < 0.5 {
		weaknesses = append(weaknesses, "low semantic relevance to retrieved context")
	}
	if score.QueryCoverage < 0.5 {
		weaknesses = append(weaknesses, "missing key query terms in answer")
	}
	if score.AnswerCompleteness < 0.5 {
		weaknesses = append(weaknesses, "answer lacks detail or structure")
	}
	if score.ContextUsage < 0.3 {
		weaknesses = append(weaknesses, "retrieved context not effectively used")
	}

	return weaknesses
}
