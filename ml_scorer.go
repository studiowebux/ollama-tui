package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"

	onnxruntime "github.com/yalue/onnxruntime_go"
)

// MLScorer uses ONNX model for quality prediction
type MLScorer struct {
	session      *onnxruntime.DynamicAdvancedSession
	metadata     *ModelMetadata
	isAvailable  bool
	fallbackMode bool
}

// ModelMetadata contains feature normalization parameters
type ModelMetadata struct {
	FeatureNames []string  `json:"feature_names"`
	Mean         []float64 `json:"mean"`
	Std          []float64 `json:"std"`
}

// NewMLScorer creates a new ML-based quality scorer
// Returns (nil, error) on failure - caller must check error and handle appropriately
func NewMLScorer(modelPath, metadataPath, onnxLibPath string) (*MLScorer, error) {
	// Validate paths
	if modelPath == "" || metadataPath == "" {
		return nil, fmt.Errorf("model path and metadata path are required")
	}

	// Check if files exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", modelPath)
	}
	if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("metadata file not found: %s", metadataPath)
	}

	// Load metadata
	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata: %w", err)
	}

	var metadata ModelMetadata
	if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata JSON: %w", err)
	}

	// Set ONNX library path (use provided or platform default)
	if onnxLibPath != "" {
		onnxruntime.SetSharedLibraryPath(onnxLibPath)
	}
	// If empty, onnxruntime uses platform defaults

	// Initialize ONNX runtime
	if err := onnxruntime.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime (is onnxruntime installed?): %w", err)
	}

	// Create session
	inputNames := []string{"input"}
	outputNames := []string{"output"}
	session, err := onnxruntime.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil,
	)
	if err != nil {
		onnxruntime.DestroyEnvironment()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &MLScorer{
		session:      session,
		metadata:     &metadata,
		isAvailable:  true,
		fallbackMode: false,
	}, nil
}

// Close releases ONNX resources
func (s *MLScorer) Close() {
	if s.session != nil {
		s.session.Destroy()
	}
	if s.isAvailable {
		onnxruntime.DestroyEnvironment()
	}
}

// IsAvailable returns true if ML model is loaded and ready
func (s *MLScorer) IsAvailable() bool {
	return s.isAvailable
}

// ScoreAnswer predicts quality score using ML model
// Returns error if prediction fails - caller should handle fallback
func (s *MLScorer) ScoreAnswer(query, answer string, ragResult *RAGResult, config *Config) (*QualityScore, error) {
	if !s.isAvailable {
		return nil, fmt.Errorf("ML scorer not initialized")
	}

	// Extract features (matching Python pipeline)
	features := s.extractFeatures(query, answer, ragResult, config)

	// Normalize features
	normalizedFeatures := s.normalizeFeatures(features)

	// Run ONNX inference
	score, err := s.predict(normalizedFeatures)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// Build quality score result
	qualityScore := &QualityScore{
		OverallScore: score,
		Details:      make(map[string]float64),
	}

	// Add feature values for debugging
	for i, name := range s.metadata.FeatureNames {
		qualityScore.Details[name] = features[i]
	}
	qualityScore.Details["ml_prediction"] = score

	return qualityScore, nil
}

// extractFeatures extracts 15 features matching Python pipeline
func (s *MLScorer) extractFeatures(query, answer string, ragResult *RAGResult, config *Config) []float64 {
	features := make([]float64, 15)

	// Metadata features (4)
	if ragResult.ContextUsed {
		features[0] = 1.0
	} else {
		features[0] = 0.0
	}
	features[1] = float64(ragResult.ContextsUsed)
	features[2] = float64(config.VectorTopK)
	features[3] = config.VectorSimilarity

	// Text-based features (6)
	features[4] = float64(len(query))                                  // query_length
	features[5] = float64(len(answer))                                 // answer_length
	features[6] = float64(len(answer)) / math.Max(float64(len(query)), 1.0) // answer_query_ratio
	features[7] = s.calculateQueryCoverage(query, answer)              // query_coverage
	features[8] = s.calculateAnswerCompleteness(answer)                // answer_completeness

	// Word-level features (2)
	queryWords := len(strings.Fields(query))
	answerWords := len(strings.Fields(answer))
	features[9] = float64(queryWords)                                  // query_word_count
	features[10] = float64(answerWords)                                // answer_word_count
	features[11] = float64(answerWords) / math.Max(float64(ragResult.ContextsUsed), 1.0) // words_per_chunk

	// Structural features (3)
	if strings.Contains(answer, "\n\n") {
		features[12] = 1.0 // has_paragraphs
	}
	if strings.Contains(answer, "```") {
		features[13] = 1.0 // has_code_blocks
	}
	if strings.Contains(answer, "- ") || strings.Contains(answer, "* ") {
		features[14] = 1.0 // has_lists
	}

	return features
}

// calculateQueryCoverage calculates percentage of query terms in answer
func (s *MLScorer) calculateQueryCoverage(query, answer string) float64 {
	stopwords := map[string]bool{
		"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
		"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
		"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
		"that": true, "the": true, "to": true, "was": true, "will": true, "with": true,
		"how": true, "what": true, "when": true, "where": true, "who": true, "why": true,
	}

	// Extract significant words
	wordRegex := regexp.MustCompile(`\b\w+\b`)
	queryWords := wordRegex.FindAllString(strings.ToLower(query), -1)

	var significantWords []string
	for _, word := range queryWords {
		if len(word) > 2 && !stopwords[word] {
			significantWords = append(significantWords, word)
		}
	}

	if len(significantWords) == 0 {
		return 0.0
	}

	answerLower := strings.ToLower(answer)
	matched := 0
	for _, word := range significantWords {
		if strings.Contains(answerLower, word) {
			matched++
		}
	}

	return float64(matched) / float64(len(significantWords))
}

// calculateAnswerCompleteness calculates completeness score based on length and structure
func (s *MLScorer) calculateAnswerCompleteness(answer string) float64 {
	length := len(answer)

	var lengthScore float64
	if length < 50 {
		lengthScore = 0.3
	} else if length < 150 {
		lengthScore = 0.6
	} else if length < 500 {
		lengthScore = 0.8
	} else {
		lengthScore = 1.0
	}

	structureBonus := 0.0
	if strings.Contains(answer, "\n\n") {
		structureBonus += 0.1
	}
	if strings.Contains(answer, "```") || strings.Contains(answer, "- ") || strings.Contains(answer, "* ") {
		structureBonus += 0.1
	}

	return math.Min(1.0, lengthScore+structureBonus)
}

// normalizeFeatures applies mean/std normalization
func (s *MLScorer) normalizeFeatures(features []float64) []float64 {
	normalized := make([]float64, len(features))
	for i := range features {
		std := s.metadata.Std[i]
		if std == 0 {
			std = 1.0
		}
		normalized[i] = (features[i] - s.metadata.Mean[i]) / std
	}
	return normalized
}

// predict runs ONNX inference
func (s *MLScorer) predict(features []float64) (float64, error) {
	// Convert to float32 for ONNX
	inputData := make([]float32, len(features))
	for i, v := range features {
		inputData[i] = float32(v)
	}

	// Create input tensor
	inputShape := onnxruntime.NewShape(1, int64(len(features)))
	inputTensor, err := onnxruntime.NewTensor(inputShape, inputData)
	if err != nil {
		return 0, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor
	outputShape := onnxruntime.NewShape(1, 1)
	outputTensor, err := onnxruntime.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return 0, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference
	err = s.session.Run(
		[]onnxruntime.ArbitraryTensor{inputTensor},
		[]onnxruntime.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return 0, fmt.Errorf("inference failed: %w", err)
	}

	// Extract result
	outputData := outputTensor.GetData()
	if len(outputData) == 0 {
		return 0, fmt.Errorf("empty output from model")
	}

	return float64(outputData[0]), nil
}
