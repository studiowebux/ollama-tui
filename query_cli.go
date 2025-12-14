package main

import (
	"bufio"
	"fmt"
	"ollamatui/cmd"
	"os"
	"strconv"
	"strings"
	"time"
)

func init() {
	// Register the query runner
	cmd.QueryRunner = runQueryCommand

	// Register completion functions
	cmd.CompleteQueryModels = completeChatModels    // Reuse from import
	cmd.CompleteQueryProjects = completeProjects    // Reuse from import
	cmd.RegisterQueryCompletions()
}

func runQueryCommand() {
	// Load config
	config, err := LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	// Initialize ML scorer if explicitly enabled in config
	var mlScorer *MLScorer
	if config.MLEnableScoring && config.MLModelPath != "" && config.MLMetadataPath != "" {
		mlScorer, err = NewMLScorer(config.MLModelPath, config.MLMetadataPath, config.MLOnnxLibPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to load ML model, using heuristic scoring: %v\n", err)
			mlScorer = nil
		}
	}

	// Use config values if not specified
	if cmd.QueryProject == "" {
		cmd.QueryProject = config.CurrentProject
	}
	if cmd.QueryModel == "" {
		cmd.QueryModel = config.Model
	}

	// Initialize project manager
	pm, err := NewProjectManager()
	if err != nil {
		fmt.Printf("Error initializing project manager: %v\n", err)
		os.Exit(1)
	}

	// Verify project exists
	project := pm.GetProject(cmd.QueryProject)
	if project == nil {
		fmt.Printf("Error: Project '%s' does not exist\n", cmd.QueryProject)
		fmt.Println("\nAvailable projects:")
		for _, p := range pm.ListProjects() {
			fmt.Printf("  - %s (%s)\n", p.ID, p.Name)
		}
		os.Exit(1)
	}

	// Initialize VectorDB
	vectorDB, err := NewVectorDB(pm, cmd.QueryProject)
	if err != nil {
		fmt.Printf("Error initializing vector DB: %v\n", err)
		os.Exit(1)
	}

	// Initialize Ollama client
	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = config.Endpoint
	}
	client := NewOllamaClient(endpoint)

	// Test connection and verify model
	models, err := client.ListModels()
	if err != nil {
		fmt.Printf("Error connecting to Ollama at %s: %v\n", endpoint, err)
		fmt.Println("Make sure Ollama is running and accessible.")
		os.Exit(1)
	}

	modelExists := false
	for _, m := range models {
		if m == cmd.QueryModel {
			modelExists = true
			break
		}
	}

	if !modelExists {
		fmt.Printf("Error: Model '%s' not found\n", cmd.QueryModel)
		fmt.Printf("\nAvailable models:\n")
		for _, m := range models {
			fmt.Printf("  - %s\n", m)
		}
		fmt.Printf("\nPull the model with: ollama pull %s\n", cmd.QueryModel)
		os.Exit(1)
	}

	if cmd.QueryVerbose {
		fmt.Printf("Project: %s\n", project.Name)
		fmt.Printf("Model: %s\n", cmd.QueryModel)
		fmt.Printf("Query: %s\n", cmd.QueryPrompt)
		fmt.Println()
	}

	// Create RAG engine
	ragEngine := NewRAGEngine(client, vectorDB, config)

	// Retrieve relevant context
	ragResult, err := ragEngine.RetrieveContext(cmd.QueryPrompt)
	if err != nil {
		fmt.Printf("Error retrieving context: %v\n", err)
		os.Exit(1)
	}

	if cmd.QueryVerbose {
		fmt.Println("=== Vector Search Results ===")
		fmt.Println(ragResult.DebugInfo)
		fmt.Println()
	}

	// Build messages with context
	var messages []ChatMessage

	if ragResult.ContextUsed {
		// Add system instruction BEFORE context
		messages = append(messages, ChatMessage{
			Role: "system",
			Content: `You must answer questions directly and concisely using the provided context.

CRITICAL RULES - MUST FOLLOW:
1. If user says "X words max" or "X words" - YOUR ANSWER MUST BE EXACTLY THAT LENGTH OR SHORTER
2. Answer ONLY the specific question - do not add extra information
3. Do not write essays or long explanations
4. Do not ignore word limits
5. Be direct and brief

EXAMPLE:
Question: "who is X, 10 words max"
Good answer: "X is an ancient entity that created existence."
Bad answer: Long paragraph explaining everything about X

Now use this context to answer the user's question:`,
		})

		// Add context as separate message
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: ragResult.Context,
		})
	} else {
		// No context available - add basic instruction
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: "Answer the user's question directly and concisely. If they specify a word limit, you MUST follow it exactly. Do not write long explanations when brevity is requested.",
		})
	}

	// Add user query with reinforced constraints
	userPrompt := cmd.QueryPrompt

	// If query contains word limit, make it VERY explicit
	if strings.Contains(strings.ToLower(userPrompt), "words") &&
	   (strings.Contains(strings.ToLower(userPrompt), "max") ||
	    strings.Contains(strings.ToLower(userPrompt), "limit")) {
		userPrompt = userPrompt + "\n\nIMPORTANT: Your answer must be brief and respect the word limit specified above. Count your words carefully."
	}

	messages = append(messages, ChatMessage{
		Role:    "user",
		Content: userPrompt,
	})

	// Generate initial response (non-streaming for CLI)
	if cmd.QueryVerbose {
		fmt.Println("=== Generating Answer ===")
	}

	response, err := client.Chat(cmd.QueryModel, messages)
	if err != nil {
		fmt.Printf("Error generating response: %v\n", err)
		os.Exit(1)
	}

	// Perform refinement if enabled
	finalAnswer := response
	var refinementResult *RefinementResult

	// Skip refinement for queries with explicit word limits (they want brief answers)
	hasWordLimit := strings.Contains(strings.ToLower(cmd.QueryPrompt), "words") &&
		(strings.Contains(strings.ToLower(cmd.QueryPrompt), "max") ||
		 strings.Contains(strings.ToLower(cmd.QueryPrompt), "limit") ||
		 strings.Contains(strings.ToLower(cmd.QueryPrompt), "brief") ||
		 strings.Contains(strings.ToLower(cmd.QueryPrompt), "short"))

	if config.EnableRefinement && !hasWordLimit {
		refinementEngine := NewRefinementEngine(client, ragEngine, config, mlScorer)

		progressChan := make(chan string, 10)
		done := make(chan bool)

		if cmd.QueryVerbose {
			fmt.Println("\n=== Refinement Process ===")
			go func() {
				for msg := range progressChan {
					fmt.Printf("  %s\n", msg)
				}
				done <- true
			}()
		} else {
			go func() {
				for range progressChan {
				}
				done <- true
			}()
		}

		refinementResult, err = refinementEngine.RefineAnswer(cmd.QueryPrompt, response, ragResult, cmd.QueryModel, progressChan)
		close(progressChan)
		<-done

		if err != nil {
			if cmd.QueryVerbose {
				fmt.Printf("  Warning: Refinement failed: %v\n", err)
			}
		} else {
			finalAnswer = refinementResult.FinalAnswer
		}
	}

	// Output the final answer
	if cmd.QueryVerbose {
		fmt.Println("\n=== Final Answer ===")
	}
	fmt.Println(strings.TrimSpace(finalAnswer))

	if cmd.QueryVerbose {
		fmt.Println()
		fmt.Printf("Context used: %d chunks\n", ragResult.ContextsUsed)
		fmt.Printf("Total results found: %d\n", ragResult.ResultsCount)

		if refinementResult != nil && refinementResult.WasRefined {
			fmt.Println("\nRefinement Summary:")
			fmt.Printf("  Initial quality: %.2f\n", refinementResult.InitialScore.OverallScore)
			fmt.Printf("  Final quality: %.2f\n", refinementResult.FinalScore.OverallScore)
			fmt.Printf("  Passes performed: %d\n", refinementResult.PassesPerformed)
		}
	}

	// Prompt for rating if requested
	if cmd.QueryRate {
		rating, err := promptForRating()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading rating: %v\n", err)
		} else if rating > 0 {
			// Initialize storage for saving rating
			userStorage, err := NewStorage(pm, cmd.QueryProject)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error initializing storage: %v\n", err)
			} else {
				err = saveQueryRating(userStorage, cmd.QueryPrompt, finalAnswer, rating, ragResult, config)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error saving rating: %v\n", err)
				} else {
					fmt.Printf("\n✓ Rating saved: %s (%d/5)\n", strings.Repeat("⭐", rating)+strings.Repeat("☆", 5-rating), rating)
				}
			}
		}
	}
}

// promptForRating prompts user for a 1-5 star rating
func promptForRating() (int, error) {
	fmt.Print("\nRate this answer (1-5 stars, or 0 to skip): ")
	reader := bufio.NewReader(os.Stdin)
	input, err := reader.ReadString('\n')
	if err != nil {
		return 0, err
	}

	input = strings.TrimSpace(input)
	if input == "" || input == "0" {
		return 0, nil
	}

	rating, err := strconv.Atoi(input)
	if err != nil || rating < 1 || rating > 5 {
		fmt.Println("Invalid rating. Skipping.")
		return 0, nil
	}

	return rating, nil
}

// saveQueryRating saves a rating for a CLI query
func saveQueryRating(storage *Storage, query, answer string, rating int, ragResult *RAGResult, config *Config) error {
	// Create a temporary chat to store the rating
	chat := &Chat{
		ID:        fmt.Sprintf("query_%d", time.Now().Unix()),
		Title:     "CLI Query",
		Model:     config.Model,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Messages:  []Message{},
	}

	// Add user query
	chat.Messages = append(chat.Messages, Message{
		Role:      "user",
		Content:   query,
		Timestamp: time.Now(),
	})

	// Add assistant answer with rating
	chat.Messages = append(chat.Messages, Message{
		Role:      "assistant",
		Content:   answer,
		Timestamp: time.Now(),
		Rating: &Rating{
			Score:            rating,
			Timestamp:        time.Now(),
			Query:            query,
			ContextUsed:      ragResult.ContextUsed,
			ContextChunks:    ragResult.ContextsUsed,
			Model:            config.Model,
			VectorTopK:       config.VectorTopK,
			VectorSimilarity: config.VectorSimilarity,
		},
	})

	return storage.SaveChat(chat)
}
