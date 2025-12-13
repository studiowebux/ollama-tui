package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	projectManager, err := NewProjectManager()
	if err != nil {
		fmt.Printf("Error initializing project manager: %v\n", err)
		os.Exit(1)
	}

	config, err := LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	storage, err := NewStorage(projectManager, config.CurrentProject)
	if err != nil {
		fmt.Printf("Error initializing storage: %v\n", err)
		os.Exit(1)
	}

	vectorDB, err := NewVectorDB(projectManager, config.CurrentProject)
	if err != nil {
		fmt.Printf("Error initializing vector DB: %v\n", err)
		os.Exit(1)
	}

	client := NewOllamaClient(config.Endpoint)

	p := tea.NewProgram(
		initialModel(storage, client, config, vectorDB, projectManager),
		tea.WithAltScreen(),
	)

	if _, err := p.Run(); err != nil {
		fmt.Printf("Error running program: %v\n", err)
		os.Exit(1)
	}
}
