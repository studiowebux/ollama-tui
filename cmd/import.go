package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var (
	ImportProject    string
	ImportChatModel  string
	ImportEmbedModel string
	ImportStrategy   string
	ImportForce      bool
	ImportVerbose    bool
	ImportPath       string
)

// ImportRunner is the function that actually runs the import (defined in main package)
var ImportRunner func()

var importCmd = &cobra.Command{
	Use:   "import <file_or_directory_path>",
	Short: "Import documents into the vector database",
	Long: `Import markdown, code, and other supported documents into the vector database.
The documents will be chunked, embedded, and made searchable.`,
	Args: cobra.ExactArgs(1),
	ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		// File/directory completion
		return nil, cobra.ShellCompDirectiveDefault
	},
	Run: func(cmd *cobra.Command, args []string) {
		ImportPath = args[0]
		if ImportRunner != nil {
			ImportRunner()
		} else {
			fmt.Println("Error: Import runner not initialized")
			os.Exit(1)
		}
	},
}

// Completion functions (defined in main package)
var (
	CompleteProjects    func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
	CompleteChatModels  func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
	CompleteEmbedModels func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
	CompleteStrategies  func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective)
)

func init() {
	importCmd.Flags().StringVar(&ImportProject, "project", "", "Target project (default: current project from config)")
	importCmd.Flags().StringVar(&ImportChatModel, "chat-model", "", "Model for generating summaries (default: from config)")
	importCmd.Flags().StringVar(&ImportEmbedModel, "embed-model", "", "Model for embeddings (default: from config)")
	importCmd.Flags().StringVar(&ImportStrategy, "strategy", "all", "Chunking strategy (use tab completion to see all)")
	importCmd.Flags().BoolVar(&ImportForce, "force", false, "Re-import already imported files")
	importCmd.Flags().BoolVar(&ImportVerbose, "verbose", false, "Show detailed progress")
}

// RegisterCompletions registers the completion functions (called from main package)
func RegisterCompletions() {
	if CompleteProjects != nil {
		importCmd.RegisterFlagCompletionFunc("project", CompleteProjects)
	}
	if CompleteChatModels != nil {
		importCmd.RegisterFlagCompletionFunc("chat-model", CompleteChatModels)
	}
	if CompleteEmbedModels != nil {
		importCmd.RegisterFlagCompletionFunc("embed-model", CompleteEmbedModels)
	}
	if CompleteStrategies != nil {
		importCmd.RegisterFlagCompletionFunc("strategy", CompleteStrategies)
	}
}
