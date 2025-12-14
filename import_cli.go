package main

import (
	"fmt"
	"ollamatui/cmd"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

func init() {
	// Register the import runner
	cmd.ImportRunner = runImportCommand

	// Register completion functions
	cmd.CompleteProjects = completeProjects
	cmd.CompleteChatModels = completeChatModels
	cmd.CompleteEmbedModels = completeEmbedModels
	cmd.CompleteStrategies = completeStrategies
	cmd.RegisterCompletions()
}

func runImportCommand() {
	targetPath := cmd.ImportPath

	// Validate path exists
	info, err := os.Stat(targetPath)
	if err != nil {
		fmt.Printf("Error: Path does not exist: %s\n", targetPath)
		os.Exit(1)
	}

	// Load config
	config, err := LoadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	// Use config values if not specified
	if cmd.ImportProject == "" {
		cmd.ImportProject = config.CurrentProject
	}
	if cmd.ImportChatModel == "" {
		cmd.ImportChatModel = config.Model
	}
	if cmd.ImportEmbedModel == "" {
		cmd.ImportEmbedModel = config.VectorModel
	}

	// Initialize project manager
	pm, err := NewProjectManager()
	if err != nil {
		fmt.Printf("Error initializing project manager: %v\n", err)
		os.Exit(1)
	}

	// Verify project exists
	project := pm.GetProject(cmd.ImportProject)
	if project == nil {
		fmt.Printf("Error: Project '%s' does not exist\n", cmd.ImportProject)
		fmt.Println("\nAvailable projects:")
		for _, p := range pm.ListProjects() {
			fmt.Printf("  - %s (%s)\n", p.ID, p.Name)
		}
		os.Exit(1)
	}

	// Initialize VectorDB
	vectorDB, err := NewVectorDB(pm, cmd.ImportProject)
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

	// Test connection and verify models
	models, err := client.ListModels()
	if err != nil {
		fmt.Printf("Error connecting to Ollama at %s: %v\n", endpoint, err)
		fmt.Println("Make sure Ollama is running and accessible.")
		os.Exit(1)
	}

	// Verify models exist
	chatModelExists := false
	embedModelExists := false
	for _, m := range models {
		if m == cmd.ImportChatModel {
			chatModelExists = true
		}
		if m == cmd.ImportEmbedModel {
			embedModelExists = true
		}
	}

	if !chatModelExists {
		fmt.Printf("Error: Chat model '%s' not found\n", cmd.ImportChatModel)
		fmt.Printf("\nAvailable chat models:\n")
		for _, m := range models {
			if !isEmbedModel(m) {
				fmt.Printf("  - %s\n", m)
			}
		}
		fmt.Printf("\nPull the model with: ollama pull %s\n", cmd.ImportChatModel)
		os.Exit(1)
	}

	if !embedModelExists {
		fmt.Printf("Error: Embed model '%s' not found\n", cmd.ImportEmbedModel)
		fmt.Printf("\nAvailable embed models:\n")
		for _, m := range models {
			if isEmbedModel(m) {
				fmt.Printf("  - %s\n", m)
			}
		}
		fmt.Printf("\nPull the model with: ollama pull %s\n", cmd.ImportEmbedModel)
		os.Exit(1)
	}

	// Print header
	fmt.Println("╔════════════════════════════════════════════════════╗")
	fmt.Println("║           Document Import to VectorDB              ║")
	fmt.Println("╚════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("Project: %s\n", project.Name)
	fmt.Printf("Chat Model: %s\n", cmd.ImportChatModel)
	fmt.Printf("Embed Model: %s\n", cmd.ImportEmbedModel)
	fmt.Printf("Path: %s\n", targetPath)
	fmt.Println()

	// Create document importer
	basePath := targetPath
	if !info.IsDir() {
		basePath = filepath.Dir(targetPath)
	}
	importer := NewDocumentImporter(client, vectorDB, basePath)

	// Collect files to process
	var filesToProcess []string
	if info.IsDir() {
		fmt.Println("Scanning directory...")
		files, err := importer.ScanDirectory(targetPath)
		if err != nil {
			fmt.Printf("Error scanning directory: %v\n", err)
			os.Exit(1)
		}
		filesToProcess = files
	} else {
		filesToProcess = []string{targetPath}
	}

	if len(filesToProcess) == 0 {
		fmt.Println("No supported files found to import.")
		fmt.Println("\nSupported extensions:")
		for ext, docType := range importer.SupportedExtensions() {
			fmt.Printf("  %s (%s)\n", ext, docType)
		}
		os.Exit(0)
	}

	fmt.Printf("Found %d files to process\n\n", len(filesToProcess))

	// Import files
	successCount := 0
	skipCount := 0
	failCount := 0

	initialChunkCount := len(vectorDB.GetAllChunks())

	for i, filePath := range filesToProcess {
		relPath, _ := filepath.Rel(basePath, filePath)
		fmt.Printf("[%d/%d] Processing: %s\n", i+1, len(filesToProcess), relPath)

		progressChan := make(chan string, 10)
		done := make(chan bool)

		if cmd.ImportVerbose {
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

		err := importer.ImportDocumentWithStrategy(filePath, cmd.ImportChatModel, cmd.ImportEmbedModel, cmd.ImportStrategy, cmd.ImportForce, progressChan)
		close(progressChan)
		<-done

		if err != nil {
			if strings.Contains(err.Error(), "already imported") {
				fmt.Println("  ⊗ Skipped (already imported)")
				skipCount++
			} else {
				fmt.Printf("  ✗ Failed: %v\n", err)
				failCount++
			}
		} else {
			fmt.Println("  ✓ Imported")
			successCount++
		}
	}

	finalChunkCount := len(vectorDB.GetAllChunks())
	chunksCreated := finalChunkCount - initialChunkCount

	// Print summary
	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════╗")
	fmt.Println("║                Import Summary                      ║")
	fmt.Println("╚════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("Files Scanned:         %d\n", len(filesToProcess))
	fmt.Printf("Successfully Imported: %d\n", successCount)
	if skipCount > 0 {
		fmt.Printf("Already Imported:      %d\n", skipCount)
	}
	if failCount > 0 {
		fmt.Printf("Failed:                %d\n", failCount)
	}
	fmt.Printf("\nTotal Chunks Created:  %d\n", chunksCreated)
	fmt.Printf("Storage Path:          %s\n", vectorDB.dataDir)

	stats := vectorDB.GetStats()
	fmt.Printf("Total Chunks in DB:    %d\n", stats["total_chunks"])
	fmt.Println()
}

// completeProjects provides auto-completion for project names
func completeProjects(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	pm, err := NewProjectManager()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	projects := pm.ListProjects()
	completions := make([]string, 0, len(projects))
	for _, p := range projects {
		if strings.HasPrefix(p.ID, toComplete) || strings.HasPrefix(p.Name, toComplete) {
			completions = append(completions, fmt.Sprintf("%s\t%s", p.ID, p.Name))
		}
	}

	return completions, cobra.ShellCompDirectiveNoFileComp
}

// completeChatModels provides auto-completion for chat models
func completeChatModels(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	config, err := LoadConfig()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = config.Endpoint
	}

	client := NewOllamaClient(endpoint)
	models, err := client.ListModels()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	completions := make([]string, 0)
	for _, m := range models {
		if !isEmbedModel(m) && strings.HasPrefix(m, toComplete) {
			completions = append(completions, m)
		}
	}

	return completions, cobra.ShellCompDirectiveNoFileComp
}

// completeEmbedModels provides auto-completion for embed models
func completeEmbedModels(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	config, err := LoadConfig()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	endpoint := os.Getenv("OLLAMA_ENDPOINT")
	if endpoint == "" {
		endpoint = config.Endpoint
	}

	client := NewOllamaClient(endpoint)
	models, err := client.ListModels()
	if err != nil {
		return nil, cobra.ShellCompDirectiveError
	}

	completions := make([]string, 0)
	for _, m := range models {
		if isEmbedModel(m) && strings.HasPrefix(m, toComplete) {
			completions = append(completions, m)
		}
	}

	return completions, cobra.ShellCompDirectiveNoFileComp
}

// isEmbedModel determines if a model is an embedding model based on naming patterns
func isEmbedModel(modelName string) bool {
	embedPatterns := []string{
		"embed",
		"nomic",
		"mxbai",
		"all-minilm",
		"bge-",
	}

	lowerName := strings.ToLower(modelName)
	for _, pattern := range embedPatterns {
		if strings.Contains(lowerName, pattern) {
			return true
		}
	}

	return false
}

// completeStrategies provides auto-completion for chunking strategies
func completeStrategies(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	strategies := []string{
		"all\tApply ALL 14 strategies",

		// Basic strategies
		"entity_sheet\tExtract characters, locations, items",
		"who_what_why\tStructured Q&A extraction",
		"keyword\tKeyword-based indexing",
		"sentence\tSentence-level chunks",
		"full_qa\tGenerate Q&A pairs",
		"document_section\tMarkdown section splitting",
		"code_snippet\tCode extraction with summaries",

		// Advanced narrative strategies
		"relationship_mapping\tExtract entity relationships",
		"timeline\tChronological event extraction",
		"conflict_plot\tNarrative conflicts and plot points",
		"rule_mechanic\tGame rules, magic systems, world mechanics",

		// Project planning strategies
		"project_planning\tExtract goals, scope, risks, constraints",
		"requirements\tFunctional/non-functional requirements",
		"task_breakdown\tActionable tasks and work items",
	}

	completions := make([]string, 0)
	for _, s := range strategies {
		if strings.HasPrefix(s, toComplete) {
			completions = append(completions, s)
		}
	}

	return completions, cobra.ShellCompDirectiveNoFileComp
}
