package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var (
	QueryPrompt  string
	QueryModel   string
	QueryProject string
	QueryVerbose bool
	QueryRate    bool // Prompt for rating after answer
)

// QueryRunner is the function that actually runs the query (defined in main package)
var QueryRunner func()

var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Query the vector database and get an AI-generated answer",
	Long: `Query the vector database with a prompt and get an AI-generated answer.
The system will search for relevant context and generate a response using the specified model.`,
	Run: func(cmd *cobra.Command, args []string) {
		if QueryPrompt == "" {
			fmt.Println("Error: --prompt is required")
			os.Exit(1)
		}
		if QueryRunner != nil {
			QueryRunner()
		} else {
			fmt.Println("Error: Query runner not initialized")
			os.Exit(1)
		}
	},
}

// Completion functions are injected from main package
var (
	CompleteQueryModels   func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
	CompleteQueryProjects func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
)

func init() {
	queryCmd.Flags().StringVarP(&QueryPrompt, "prompt", "p", "", "Query prompt (required)")
	queryCmd.Flags().StringVarP(&QueryModel, "model", "m", "", "Model to use (default: from config)")
	queryCmd.Flags().StringVar(&QueryProject, "project", "", "Target project (default: current project from config)")
	queryCmd.Flags().BoolVarP(&QueryVerbose, "verbose", "v", false, "Show detailed debug information")
	queryCmd.Flags().BoolVarP(&QueryRate, "rate", "r", false, "Prompt to rate the answer (for ML training)")

	queryCmd.MarkFlagRequired("prompt")
}

// RegisterQueryCompletions registers the completion functions (called from main package)
func RegisterQueryCompletions() {
	if CompleteQueryModels != nil {
		queryCmd.RegisterFlagCompletionFunc("model", CompleteQueryModels)
	}
	if CompleteQueryProjects != nil {
		queryCmd.RegisterFlagCompletionFunc("project", CompleteQueryProjects)
	}
}
