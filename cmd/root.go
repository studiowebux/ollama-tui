package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

const Version = "0.0.1"

var versionFlag bool

var rootCmd = &cobra.Command{
	Use:   "ollamatui",
	Short: "OllamaTUI - Terminal UI for Ollama with RAG capabilities",
	Long:  `A terminal user interface for Ollama with document import and vector search capabilities.`,
	Run: func(cmd *cobra.Command, args []string) {
		if versionFlag {
			fmt.Printf("ollamatui version %s\n", Version)
			return
		}
		// When no subcommand is specified, launch the TUI
		LaunchTUI()
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	// Add flags
	rootCmd.Flags().BoolVarP(&versionFlag, "version", "v", false, "Print version information")

	// Add subcommands
	rootCmd.AddCommand(importCmd)
}
