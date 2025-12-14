package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var (
	ExportRatingsProject string
	ExportRatingsOutput  string
)

// ExportRatingsRunner is the function that actually runs the export (defined in main package)
var ExportRatingsRunner func()

var exportRatingsCmd = &cobra.Command{
	Use:   "export-ratings",
	Short: "Export ratings to JSONL for ML training",
	Long: `Export all rated conversations to JSONL format for machine learning training.
Each line contains a rating with query, answer, score, and metadata.`,
	Run: func(cmd *cobra.Command, args []string) {
		if ExportRatingsOutput == "" {
			fmt.Println("Error: --output is required")
			os.Exit(1)
		}
		if ExportRatingsRunner != nil {
			ExportRatingsRunner()
		} else {
			fmt.Println("Error: Export ratings runner not initialized")
			os.Exit(1)
		}
	},
}

// Completion functions are injected from main package
var CompleteExportProjects func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)

func init() {
	exportRatingsCmd.Flags().StringVar(&ExportRatingsProject, "project", "", "Target project (default: current project from config)")
	exportRatingsCmd.Flags().StringVarP(&ExportRatingsOutput, "output", "o", "", "Output file path (required)")

	exportRatingsCmd.MarkFlagRequired("output")
}

// RegisterExportRatingsCompletions registers the completion functions
func RegisterExportRatingsCompletions() {
	if CompleteExportProjects != nil {
		exportRatingsCmd.RegisterFlagCompletionFunc("project", CompleteExportProjects)
	}
}
