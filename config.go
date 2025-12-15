package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type Config struct {
	Endpoint              string  `json:"endpoint"`
	Model                 string  `json:"model"`
	SummaryPrompt         string  `json:"summary_prompt"`
	CurrentProject        string  `json:"current_project"`
	VectorEnabled         bool    `json:"vector_enabled"`
	VectorModel           string  `json:"vector_model"`
	VectorTopK            int     `json:"vector_top_k"`
	VectorSimilarity      float64 `json:"vector_similarity_threshold"`
	VectorDebug           bool    `json:"vector_debug"`
	VectorExtractMetadata bool    `json:"vector_extract_metadata"`      // Extract metadata during vectorization
	VectorEnhanceQuery    bool    `json:"vector_enhance_query"`         // Enhance queries at message-send time (slow)
	VectorIncludeRelated  bool    `json:"vector_include_related"`
	VectorLightMode       bool    `json:"vector_light_mode"`       // Skip heavy extractions for slow systems
	VectorFuzzyThreshold  int     `json:"vector_fuzzy_threshold"`  // 0=disabled, 1-3=max edit distance for fuzzy matching
	VectorCompressContext bool    `json:"vector_compress_context"` // Use LLM to compress context to key facts (slower but more accurate)

	// Iterative refinement settings
	EnableRefinement           bool    `json:"enable_refinement"`             // Enable iterative refinement
	MaxRefinementPasses        int     `json:"max_refinement_passes"`         // Max number of refinement iterations
	RefinementQualityThreshold float64 `json:"refinement_quality_threshold"`  // Trigger refinement if quality < threshold

	// ML quality prediction settings
	MLModelPath      string `json:"ml_model_path"`       // Path to ONNX model file (empty = use heuristic)
	MLMetadataPath   string `json:"ml_metadata_path"`    // Path to model metadata JSON
	MLOnnxLibPath    string `json:"ml_onnx_lib_path"`    // Path to ONNX runtime library (empty = platform default)
	MLEnableScoring  bool   `json:"ml_enable_scoring"`   // Enable ML-based quality scoring (false = always use heuristic)
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	configDir := filepath.Join(home, ".ollamatui")
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
		CurrentProject:        "default",
		VectorEnabled:         true,
		VectorModel:           "nomic-embed-text",
		VectorTopK:            5,     // Default: 5 chunks (reasonable for most queries)
		VectorSimilarity:      0.7,
		VectorDebug:           false,
		VectorExtractMetadata: true,
		VectorEnhanceQuery:    false, // Disabled by default for speed
		VectorIncludeRelated:  false,
		VectorLightMode:       false,
		VectorFuzzyThreshold:  2, // Default: edit distance <= 2 for fuzzy matching
		VectorCompressContext: false, // Disabled by default (adds LLM call overhead)

		// Refinement defaults (prioritize quality)
		EnableRefinement:           true,
		MaxRefinementPasses:        2,
		RefinementQualityThreshold: 0.6,

		// ML defaults (disabled by default, no hardcoded paths)
		MLModelPath:     "",    // Empty = use heuristic
		MLMetadataPath:  "",    // Empty = use heuristic
		MLOnnxLibPath:   "",    // Empty = platform default
		MLEnableScoring: false, // Explicit opt-in required
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

	// Validate and fix project reference
	if err := config.ValidateAndFix(); err != nil {
		return nil, err
	}

	return config, nil
}

// ValidateAndFix ensures config references valid projects
func (c *Config) ValidateAndFix() error {
	// Initialize project manager to check if project exists
	pm, err := NewProjectManager()
	if err != nil {
		return err
	}

	// Check if current project exists
	project := pm.GetProject(c.CurrentProject)
	if project == nil {
		// Project doesn't exist - find first valid project or create default
		projects := pm.ListProjects()
		if len(projects) > 0 {
			// Use first available project
			c.CurrentProject = projects[0].ID
		} else {
			// No projects exist - create default
			defaultProj := &Project{
				ID:   "default",
				Name: "Default Project",
			}
			if err := pm.CreateProject(defaultProj); err != nil {
				return err
			}
			c.CurrentProject = defaultProj.ID
		}
		// Save corrected config
		c.Save()
	}

	return nil
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
