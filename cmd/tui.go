package cmd

// TUILauncher is the function that launches the TUI (defined in main package)
var TUILauncher func()

// LaunchTUI starts the terminal UI
func LaunchTUI() {
	if TUILauncher != nil {
		TUILauncher()
	}
}
