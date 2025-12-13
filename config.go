package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type Config struct {
	Endpoint      string `json:"endpoint"`
	Model         string `json:"model"`
	SummaryPrompt string `json:"summary_prompt"`
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	configDir := filepath.Join(home, ".ollama-ui")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return "", err
	}
	return filepath.Join(configDir, "config.json"), nil
}

func LoadConfig() (*Config, error) {
	path, err := configPath()
	if err != nil {
		return nil, err
	}

	config := &Config{
		Endpoint: "http://localhost:11434",
		Model:    "llama2",
		SummaryPrompt: `You are tasked with summarizing a conversation to preserve essential information while reducing context size.

REQUIREMENTS:
1. Extract and preserve all key decisions, conclusions, and action items
2. Maintain technical details: code snippets, commands, configurations, file paths, URLs
3. Preserve specific numbers, versions, parameters, and measurements
4. Keep error messages, warnings, and their solutions
5. Document the logical flow and reasoning behind decisions
6. Include relevant context needed to continue the conversation seamlessly

FORMAT:
- Use clear, structured markdown with headers
- Group related information together
- Be concise but complete - don't omit critical details
- Focus on facts and outcomes, not conversational filler

CONVERSATION TO SUMMARIZE:

`,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return config, nil
		}
		return nil, err
	}

	if err := json.Unmarshal(data, config); err != nil {
		return nil, err
	}

	return config, nil
}

func (c *Config) Save() error {
	path, err := configPath()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}
