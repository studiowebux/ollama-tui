package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type OllamaClient struct {
	endpoint string
	client   *http.Client
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type ChatResponse struct {
	Message ChatMessage `json:"message"`
	Done    bool        `json:"done"`
}

type ModelsResponse struct {
	Models []Model `json:"models"`
}

type Model struct {
	Name string `json:"name"`
}

type ModelInfo struct {
	ModelInfo struct {
		ParameterSize string `json:"parameter_size"`
	} `json:"model_info"`
	Details struct {
		Format            string `json:"format"`
		Family            string `json:"family"`
		Families          []string `json:"families"`
		ParameterSize     string `json:"parameter_size"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
	ModelFile string `json:"modelfile"`
}

type ModelShowRequest struct {
	Name string `json:"name"`
}

type ModelShowResponse struct {
	License    string                            `json:"license"`
	Modelfile  string                            `json:"modelfile"`
	Parameters string                            `json:"parameters"`
	Template   string                            `json:"template"`
	Details    struct {
		Format            string   `json:"format"`
		Family            string   `json:"family"`
		Families          []string `json:"families"`
		ParameterSize     string   `json:"parameter_size"`
		QuantizationLevel string   `json:"quantization_level"`
	} `json:"details"`
	ModelInfo map[string]interface{} `json:"model_info"`
}

func NewOllamaClient(endpoint string) *OllamaClient {
	return &OllamaClient{
		endpoint: strings.TrimSuffix(endpoint, "/"),
		client:   &http.Client{},
	}
}

func (c *OllamaClient) ListModels() ([]string, error) {
	resp, err := c.client.Get(c.endpoint + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to list models: %s", resp.Status)
	}

	var modelsResp ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, err
	}

	models := make([]string, len(modelsResp.Models))
	for i, m := range modelsResp.Models {
		models[i] = m.Name
	}

	return models, nil
}

func (c *OllamaClient) StreamChat(model string, messages []ChatMessage, onChunk func(string) error) error {
	reqBody := ChatRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", c.endpoint+"/api/chat", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to chat: %s", resp.Status)
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chatResp ChatResponse
		if err := json.Unmarshal([]byte(line), &chatResp); err != nil {
			continue
		}

		if chatResp.Message.Content != "" {
			if err := onChunk(chatResp.Message.Content); err != nil {
				return err
			}
		}

		if chatResp.Done {
			break
		}
	}

	return scanner.Err()
}

func (c *OllamaClient) Chat(model string, messages []ChatMessage) (string, error) {
	var fullResponse strings.Builder

	err := c.StreamChat(model, messages, func(chunk string) error {
		fullResponse.WriteString(chunk)
		return nil
	})

	if err != nil {
		return "", err
	}

	return fullResponse.String(), nil
}

func (c *OllamaClient) SetEndpoint(endpoint string) {
	c.endpoint = strings.TrimSuffix(endpoint, "/")
}

func (c *OllamaClient) GenerateSummary(model, summaryPrompt string, messages []Message) (string, error) {
	var conversationText strings.Builder
	for _, msg := range messages {
		conversationText.WriteString(fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content))
	}

	chatMessages := []ChatMessage{
		{
			Role:    "user",
			Content: summaryPrompt + conversationText.String(),
		},
	}

	return c.Chat(model, chatMessages)
}

func (c *OllamaClient) EstimateTokenCount(messages []Message) int {
	totalChars := 0
	for _, msg := range messages {
		totalChars += len(msg.Content) + len(msg.Role)
	}
	return totalChars / 4
}

func (c *OllamaClient) GetModelInfo(modelName string) (*ModelShowResponse, error) {
	reqBody := ModelShowRequest{
		Name: modelName,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", c.endpoint+"/api/show", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get model info: %s", resp.Status)
	}

	var modelInfo ModelShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return nil, err
	}

	return &modelInfo, nil
}

func (c *OllamaClient) GetContextSize(modelName string) (int, error) {
	modelInfo, err := c.GetModelInfo(modelName)
	if err != nil {
		return 4096, err
	}

	contextSize := extractContextFromModelInfo(modelInfo.ModelInfo)
	if contextSize == 0 {
		contextSize = extractContextSize(modelInfo.Parameters)
	}
	if contextSize == 0 {
		contextSize = extractContextSize(modelInfo.Modelfile)
	}

	if contextSize == 0 {
		return 4096, nil
	}

	return contextSize, nil
}

func extractContextFromModelInfo(modelInfo map[string]interface{}) int {
	contextPatterns := []string{"context_length", "max_position_embeddings", "n_ctx", "max_seq_len"}

	for key, value := range modelInfo {
		keyLower := strings.ToLower(key)
		for _, pattern := range contextPatterns {
			if strings.Contains(keyLower, pattern) {
				switch v := value.(type) {
				case float64:
					return int(v)
				case int:
					return v
				case int64:
					return int(v)
				}
			}
		}
	}
	return 0
}

func extractContextSize(text string) int {
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(strings.ToLower(line), "parameter num_ctx") {
			parts := strings.Fields(line)
			if len(parts) >= 3 {
				var ctxSize int
				fmt.Sscanf(parts[2], "%d", &ctxSize)
				if ctxSize > 0 {
					return ctxSize
				}
			}
		}

		if strings.HasPrefix(line, "num_ctx") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				var ctxSize int
				fmt.Sscanf(parts[1], "%d", &ctxSize)
				if ctxSize > 0 {
					return ctxSize
				}
			}
		}
	}
	return 0
}

type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type EmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (c *OllamaClient) GenerateEmbedding(model, text string) ([]float64, error) {
	reqBody := EmbedRequest{
		Model: model,
		Input: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", c.endpoint+"/api/embed", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to generate embedding: %s", resp.Status)
	}

	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, err
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embedResp.Embeddings[0], nil
}

type ExtractionResult struct {
	Entities []string `json:"entities"`
	Topics   []string `json:"topics"`
}

type FactExtractionResult struct {
	Facts    []string `json:"facts"`
	Keywords []string `json:"keywords"`
}

type FictionalExtractionResult struct {
	WorldElement   string   `json:"world_element"`
	RuleSystem     string   `json:"rule_system"`
	CharacterRefs  []string `json:"characters"`
	LocationRefs   []string `json:"locations"`
	SearchKeywords []string `json:"search_keywords"`
	FactChunks     []string `json:"fact_chunks"`
}

type EntitySheetResult struct {
	EntityName  string            `json:"entity_name"`
	EntityType  string            `json:"entity_type"` // character, location, item, rule, etc.
	Description string            `json:"description"`
	Attributes  map[string]string `json:"attributes"`
	Keywords    []string          `json:"keywords"`
}

type StructuredQAResult struct {
	Who   string   `json:"who"`
	What  string   `json:"what"`
	Why   string   `json:"why"`
	When  string   `json:"when"`
	Where string   `json:"where"`
	How   string   `json:"how"`
	Keywords []string `json:"keywords"`
}

type KeyValuePair struct {
	Key      string   `json:"key"`
	Value    string   `json:"value"`
	Keywords []string `json:"keywords"`
}

func (c *OllamaClient) ExtractEntitiesAndTopics(model, userMsg, assistantMsg string) ([]string, []string, error) {
	prompt := fmt.Sprintf(`Extract key entities (people, places, things, concepts) and topics from this Q&A pair.
Return ONLY a JSON object with "entities" and "topics" arrays. No explanation.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, nil, err
	}

	// Try to parse JSON from response
	response = strings.TrimSpace(response)

	// Find JSON object in response
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil, nil // No extraction possible
	}

	jsonStr := response[startIdx : endIdx+1]

	var result ExtractionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil, nil // Failed to parse, return empty
	}

	return result.Entities, result.Topics, nil
}

func (c *OllamaClient) ExtractFacts(model, userMsg, assistantMsg string) ([]string, []string, error) {
	prompt := fmt.Sprintf(`Extract discrete, verifiable facts from this Q&A.
Return ONLY a JSON object with "facts" (atomic statements) and "keywords" arrays.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, nil, err
	}

	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil, nil
	}

	jsonStr := response[startIdx : endIdx+1]

	var result FactExtractionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil, nil
	}

	return result.Facts, result.Keywords, nil
}

func (c *OllamaClient) ExtractFictionalElements(model, userMsg, assistantMsg string) (*FictionalExtractionResult, error) {
	prompt := fmt.Sprintf(`Extract fictional world-building elements from this Q&A.
For EACH discrete fact, character, location, or rule mentioned, extract it separately.
Return ONLY a JSON object with:
- "world_element": overall topic being described
- "rule_system": game/world rules if applicable
- "characters": array of character names mentioned
- "locations": array of location names mentioned
- "search_keywords": array of searchable terms (names, titles, descriptors)
- "fact_chunks": array of discrete, self-contained facts that can be indexed separately

Example: If 3 NPCs are described, create 3 entries in fact_chunks, each with the NPC's full description.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil
	}

	jsonStr := response[startIdx : endIdx+1]

	var result FictionalExtractionResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil
	}

	return &result, nil
}

// DetectContentType analyzes conversation to determine content type
func (c *OllamaClient) DetectContentType(model, userMsg, assistantMsg string) (string, error) {
	prompt := fmt.Sprintf(`Classify this Q&A into ONE category:
- "fact": Factual information, real-world data, definitions
- "fictional": Stories, game rules, world-building, NPCs, creative content
- "code": Programming, technical documentation
- "dialog": General conversation, opinions, discussions

Return ONLY the category word.

Q: %s
A: %s

Category:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return "dialog", err
	}

	response = strings.TrimSpace(strings.ToLower(response))

	// Extract first word
	words := strings.Fields(response)
	if len(words) > 0 {
		category := words[0]
		// Validate category
		validCategories := map[string]bool{
			"fact": true, "fictional": true, "code": true, "dialog": true,
		}
		if validCategories[category] {
			return category, nil
		}
	}

	return "dialog", nil
}

// ExtractEntitySheets extracts structured entity information (characters, locations, etc.)
func (c *OllamaClient) ExtractEntitySheets(model, userMsg, assistantMsg string) ([]EntitySheetResult, error) {
	prompt := fmt.Sprintf(`Extract ALL entities (characters, locations, items, rules, etc.) from this Q&A.
For EACH entity, create a structured sheet with:
- entity_name: the name/title
- entity_type: character, location, item, rule, concept, etc.
- description: complete description
- attributes: key-value pairs of all attributes (appearance, abilities, stats, etc.)
- keywords: searchable terms

Return ONLY a JSON array of entity sheets. No explanation.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "[")
	endIdx := strings.LastIndex(response, "]")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil
	}

	jsonStr := response[startIdx : endIdx+1]

	var result []EntitySheetResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil
	}

	return result, nil
}

// ExtractStructuredQA extracts who/what/why/when/where/how structure
func (c *OllamaClient) ExtractStructuredQA(model, userMsg, assistantMsg string) (*StructuredQAResult, error) {
	prompt := fmt.Sprintf(`Analyze this Q&A and extract structured information:
- who: entities involved
- what: what is being described or happening
- why: purpose, reason, motivation
- when: temporal context if any
- where: location, spatial context if any
- how: mechanics, process, method
- keywords: searchable terms

Return ONLY a JSON object. Use empty string if field doesn't apply.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil
	}

	jsonStr := response[startIdx : endIdx+1]

	var result StructuredQAResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil
	}

	return &result, nil
}

// ExtractKeyValuePairs extracts key-value mappings for entity registry
func (c *OllamaClient) ExtractKeyValuePairs(model, userMsg, assistantMsg string) ([]KeyValuePair, error) {
	prompt := fmt.Sprintf(`Extract ALL key-value pairs from this Q&A.
For each entity, concept, or fact, create:
- key: the name/identifier (e.g., "The Beggar of Somewhere")
- value: the complete information about it
- keywords: searchable terms

Return ONLY a JSON array. Create as many pairs as entities/concepts exist.

Q: %s
A: %s

JSON:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	response = strings.TrimSpace(response)
	startIdx := strings.Index(response, "[")
	endIdx := strings.LastIndex(response, "]")

	if startIdx == -1 || endIdx == -1 {
		return nil, nil
	}

	jsonStr := response[startIdx : endIdx+1]

	var result []KeyValuePair
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil
	}

	return result, nil
}
