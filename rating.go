package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// RatingExportEntry represents a single rating for ML training
type RatingExportEntry struct {
	Query            string    `json:"query"`
	Answer           string    `json:"answer"`
	Rating           int       `json:"rating"`
	ContextUsed      bool      `json:"context_used"`
	ContextChunks    int       `json:"context_chunks"`
	Model            string    `json:"model"`
	VectorTopK       int       `json:"vector_top_k"`
	VectorSimilarity float64   `json:"vector_similarity"`
	Timestamp        time.Time `json:"timestamp"`
	ChatID           string    `json:"chat_id"`
	ProjectID        string    `json:"project_id"`
}

// ExportRatings exports all ratings from a project to JSONL format
func ExportRatings(pm *ProjectManager, projectID string, outputPath string) error {
	storage, err := NewStorage(pm, projectID)
	if err != nil {
		return fmt.Errorf("failed to initialize storage: %v", err)
	}

	chats, err := storage.ListChats()
	if err != nil {
		return fmt.Errorf("failed to list chats: %v", err)
	}

	// Open output file
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	exportCount := 0

	// Iterate through all chats and messages
	for _, chat := range chats {
		for i := 0; i < len(chat.Messages); i++ {
			msg := chat.Messages[i]

			// Only export rated assistant messages
			if msg.Role != "assistant" || msg.Rating == nil {
				continue
			}

			// Find the corresponding user query (should be the message before this one)
			userQuery := ""
			if i > 0 && chat.Messages[i-1].Role == "user" {
				userQuery = chat.Messages[i-1].Content
			}

			// If we couldn't find a user query, use the one stored in rating
			if userQuery == "" {
				userQuery = msg.Rating.Query
			}

			entry := RatingExportEntry{
				Query:            userQuery,
				Answer:           msg.Content,
				Rating:           msg.Rating.Score,
				ContextUsed:      msg.Rating.ContextUsed,
				ContextChunks:    msg.Rating.ContextChunks,
				Model:            msg.Rating.Model,
				VectorTopK:       msg.Rating.VectorTopK,
				VectorSimilarity: msg.Rating.VectorSimilarity,
				Timestamp:        msg.Rating.Timestamp,
				ChatID:           chat.ID,
				ProjectID:        projectID,
			}

			if err := encoder.Encode(entry); err != nil {
				return fmt.Errorf("failed to encode rating: %v", err)
			}
			exportCount++
		}
	}

	fmt.Printf("Exported %d ratings to %s\n", exportCount, outputPath)
	return nil
}

// GetRatingsStats returns statistics about ratings in a project
func GetRatingsStats(pm *ProjectManager, projectID string) (map[string]interface{}, error) {
	storage, err := NewStorage(pm, projectID)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage: %v", err)
	}

	chats, err := storage.ListChats()
	if err != nil {
		return nil, fmt.Errorf("failed to list chats: %v", err)
	}

	totalRatings := 0
	ratingCounts := map[int]int{1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	totalScore := 0
	withContext := 0
	withoutContext := 0

	for _, chat := range chats {
		for _, msg := range chat.Messages {
			if msg.Role == "assistant" && msg.Rating != nil {
				totalRatings++
				ratingCounts[msg.Rating.Score]++
				totalScore += msg.Rating.Score

				if msg.Rating.ContextUsed {
					withContext++
				} else {
					withoutContext++
				}
			}
		}
	}

	avgScore := 0.0
	if totalRatings > 0 {
		avgScore = float64(totalScore) / float64(totalRatings)
	}

	return map[string]interface{}{
		"total_ratings":     totalRatings,
		"average_score":     avgScore,
		"rating_counts":     ratingCounts,
		"with_context":      withContext,
		"without_context":   withoutContext,
	}, nil
}
