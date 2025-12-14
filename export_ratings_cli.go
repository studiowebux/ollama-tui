package main

import (
	"fmt"
	"ollamatui/cmd"
	"os"
)

func init() {
	// Register the export ratings runner
	cmd.ExportRatingsRunner = runExportRatingsCommand

	// Register completion functions
	cmd.CompleteExportProjects = completeProjects // Reuse from import
	cmd.RegisterExportRatingsCompletions()
}

func runExportRatingsCommand() {
	// Load config
	config, err := LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	// Use config values if not specified
	if cmd.ExportRatingsProject == "" {
		cmd.ExportRatingsProject = config.CurrentProject
	}

	// Initialize project manager
	pm, err := NewProjectManager()
	if err != nil {
		fmt.Printf("Error initializing project manager: %v\n", err)
		os.Exit(1)
	}

	// Verify project exists
	project := pm.GetProject(cmd.ExportRatingsProject)
	if project == nil {
		fmt.Printf("Error: Project '%s' does not exist\n", cmd.ExportRatingsProject)
		fmt.Println("\nAvailable projects:")
		for _, p := range pm.ListProjects() {
			fmt.Printf("  - %s (%s)\n", p.ID, p.Name)
		}
		os.Exit(1)
	}

	fmt.Printf("Exporting ratings from project: %s\n", project.Name)
	fmt.Printf("Output file: %s\n", cmd.ExportRatingsOutput)
	fmt.Println()

	// Export ratings
	if err := ExportRatings(pm, cmd.ExportRatingsProject, cmd.ExportRatingsOutput); err != nil {
		fmt.Printf("Error exporting ratings: %v\n", err)
		os.Exit(1)
	}

	// Show stats
	stats, err := GetRatingsStats(pm, cmd.ExportRatingsProject)
	if err != nil {
		fmt.Printf("Warning: Could not retrieve stats: %v\n", err)
		return
	}

	fmt.Println("\nRating Statistics:")
	fmt.Printf("Total ratings: %d\n", stats["total_ratings"])
	fmt.Printf("Average score: %.2f/5.0\n", stats["average_score"])
	fmt.Printf("With context: %d\n", stats["with_context"])
	fmt.Printf("Without context: %d\n", stats["without_context"])

	ratingCounts := stats["rating_counts"].(map[int]int)
	fmt.Println("\nRating distribution:")
	for i := 5; i >= 1; i-- {
		count := ratingCounts[i]
		fmt.Printf("  %d ⭐: %d\n", i, count)
	}
}
