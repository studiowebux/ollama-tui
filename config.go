package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type Config struct {
	Endpoint              string `json:"endpoint"`
	Model                 string `json:"model"`
	SummaryPrompt         string `json:"summary_prompt"`
	VectorEnabled         bool   `json:"vector_enabled"`
	VectorModel           string `json:"vector_model"`
	VectorTopK            int    `json:"vector_top_k"`
	VectorSimilarity      float64 `json:"vector_similarity_threshold"`
	VectorDebug           bool   `json:"vector_debug"`
	VectorExtractMetadata bool   `json:"vector_extract_metadata"`
	VectorIncludeRelated  bool   `json:"vector_include_related"`
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
		SummaryPrompt: "Summarize this conversation:\n- Who: names, roles, entities mentioned\n- Context: topic, purpose, domain\n- Key points: facts, opinions, decisions, technical details (code snippets, commands, file paths, URLs, numbers, versions)\n- Fictional/hypothetical: examples, scenarios, placeholders, world-building elements, rules\n- Unresolved: open questions, disagreements, errors\n- Next steps (if any)\n\nConcise. Preserve tone and intent. Maintain factual accuracy.\n\nCONVERSATION TO SUMMARIZE:\n\n",
		VectorEnabled:         true,
		VectorModel:           "nomic-embed-text",
		VectorTopK:            3,
		VectorSimilarity:      0.7,
		VectorDebug:           false,
		VectorExtractMetadata: true,
		VectorIncludeRelated:  false,
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
