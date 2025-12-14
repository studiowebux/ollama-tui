package main

import (
	"fmt"
	"ollamatui/cmd"
	"os"
	"strings"
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
		// Add system message with context
		systemMsg := ragResult.Context
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: systemMsg,
		})
	}

	// Add user query
	messages = append(messages, ChatMessage{
		Role:    "user",
		Content: cmd.QueryPrompt,
	})

	// Generate response (non-streaming for CLI)
	if cmd.QueryVerbose {
		fmt.Println("=== Generating Answer ===")
	}

	response, err := client.Chat(cmd.QueryModel, messages)
	if err != nil {
		fmt.Printf("Error generating response: %v\n", err)
		os.Exit(1)
	}

	// Output the answer
	if cmd.QueryVerbose {
		fmt.Println("\n=== Answer ===")
	}
	fmt.Println(strings.TrimSpace(response))

	if cmd.QueryVerbose {
		fmt.Println()
		fmt.Printf("Context used: %d chunks\n", ragResult.ContextsUsed)
		fmt.Printf("Total results found: %d\n", ragResult.ResultsCount)
	}
}
