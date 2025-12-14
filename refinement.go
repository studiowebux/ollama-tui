package main

import (
	"fmt"
	"strings"
)

// RefinementEngine handles iterative answer refinement
type RefinementEngine struct {
	client   *OllamaClient
	ragEngine *RAGEngine
	config   *Config
}

// NewRefinementEngine creates a new refinement engine
func NewRefinementEngine(client *OllamaClient, ragEngine *RAGEngine, config *Config) *RefinementEngine {
	return &RefinementEngine{
		client:   client,
		ragEngine: ragEngine,
		config:   config,
	}
}

// RefinementResult contains the results of the refinement process
type RefinementResult struct {
	FinalAnswer      string
	InitialScore     *QualityScore
	FinalScore       *QualityScore
	PassesPerformed  int
	WasRefined       bool
	RefinementSteps  []string
}

// RefineAnswer performs iterative refinement on an answer
func (r *RefinementEngine) RefineAnswer(query, initialAnswer string, initialRAGResult *RAGResult, model string, progressChan chan<- string) (*RefinementResult, error) {
	result := &RefinementResult{
		FinalAnswer:     initialAnswer,
		RefinementSteps: make([]string, 0),
	}

	// Calculate initial quality score
	result.InitialScore = CalculateQualityScore(query, initialAnswer, initialRAGResult)
	result.FinalScore = result.InitialScore

	if progressChan != nil {
		progressChan <- fmt.Sprintf("Initial quality score: %.2f", result.InitialScore.OverallScore)
	}

	// Check if refinement is needed
	if !r.config.EnableRefinement || !ShouldRefine(result.InitialScore, r.config.RefinementQualityThreshold) {
		if progressChan != nil {
			progressChan <- "Quality score acceptable, skipping refinement"
		}
		return result, nil
	}

	result.WasRefined = true
	currentAnswer := initialAnswer
	currentScore := result.InitialScore

	// Perform iterative refinement
	for pass := 1; pass <= r.config.MaxRefinementPasses; pass++ {
		if progressChan != nil {
			progressChan <- fmt.Sprintf("Refinement pass %d/%d...", pass, r.config.MaxRefinementPasses)
		}

		// Identify gaps and weaknesses
		weaknesses := IdentifyWeaknesses(currentScore)
		if len(weaknesses) == 0 {
			// No weaknesses identified, stop refining
			break
		}

		result.RefinementSteps = append(result.RefinementSteps, fmt.Sprintf("Pass %d: Identified weaknesses: %s", pass, strings.Join(weaknesses, ", ")))

		// Generate gap analysis query
		gapQuery, err := r.analyzeGaps(query, currentAnswer, weaknesses, model)
		if err != nil {
			if progressChan != nil {
				progressChan <- fmt.Sprintf("Gap analysis failed: %v", err)
			}
			break
		}

		result.RefinementSteps = append(result.RefinementSteps, fmt.Sprintf("Gap analysis: %s", gapQuery))

		// Perform secondary search based on gaps
		if progressChan != nil {
			progressChan <- "Searching for missing information..."
		}

		secondaryRAGResult, err := r.ragEngine.RetrieveContext(gapQuery)
		if err != nil || !secondaryRAGResult.ContextUsed {
			if progressChan != nil {
				progressChan <- "Secondary search found no new information"
			}
			break
		}

		result.RefinementSteps = append(result.RefinementSteps, fmt.Sprintf("Found %d additional chunks", secondaryRAGResult.ContextsUsed))

		// Synthesize refined answer
		if progressChan != nil {
			progressChan <- "Synthesizing refined answer..."
		}

		refinedAnswer, err := r.synthesizeAnswer(query, currentAnswer, secondaryRAGResult.Context, weaknesses, model)
		if err != nil {
			if progressChan != nil {
				progressChan <- fmt.Sprintf("Synthesis failed: %v", err)
			}
			break
		}

		// Calculate new quality score
		newScore := CalculateQualityScore(query, refinedAnswer, secondaryRAGResult)

		result.RefinementSteps = append(result.RefinementSteps, fmt.Sprintf("New quality score: %.2f (was %.2f)", newScore.OverallScore, currentScore.OverallScore))

		// Check if quality improved
		if newScore.OverallScore > currentScore.OverallScore {
			currentAnswer = refinedAnswer
			currentScore = newScore
			result.PassesPerformed++

			if progressChan != nil {
				progressChan <- fmt.Sprintf("Quality improved: %.2f → %.2f", result.InitialScore.OverallScore, newScore.OverallScore)
			}

			// Check if we've reached acceptable quality
			if !ShouldRefine(newScore, r.config.RefinementQualityThreshold) {
				if progressChan != nil {
					progressChan <- "Quality threshold reached, stopping refinement"
				}
				break
			}
		} else {
			// Quality didn't improve, stop refining
			if progressChan != nil {
				progressChan <- "Quality did not improve, stopping refinement"
			}
			break
		}
	}

	result.FinalAnswer = currentAnswer
	result.FinalScore = currentScore

	return result, nil
}

// analyzeGaps asks the LLM to identify missing information
func (r *RefinementEngine) analyzeGaps(query, currentAnswer string, weaknesses []string, model string) (string, error) {
	weaknessText := strings.Join(weaknesses, ", ")

	prompt := fmt.Sprintf(`You are analyzing an answer to identify missing information.

Original Query: %s

Current Answer:
%s

Identified Weaknesses: %s

Based on the weaknesses, what specific information is missing from this answer?
Generate a concise search query to find the missing information. Return ONLY the search query, no explanation.`, query, currentAnswer, weaknessText)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := r.client.Chat(model, messages)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(response), nil
}

// synthesizeAnswer creates a refined answer using additional context
func (r *RefinementEngine) synthesizeAnswer(query, currentAnswer, additionalContext string, weaknesses []string, model string) (string, error) {
	weaknessText := strings.Join(weaknesses, ", ")

	prompt := fmt.Sprintf(`You are refining an answer to improve its quality.

Original Query: %s

Current Answer:
%s

Identified Weaknesses: %s

Additional Context Found:
%s

Create an improved answer that addresses the weaknesses using the additional context.
Maintain the good parts of the current answer and enhance it with the new information.`, query, currentAnswer, weaknessText, additionalContext)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := r.client.Chat(model, messages)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(response), nil
}
