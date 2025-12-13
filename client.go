package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type OllamaClient struct {
	endpoint     string
	client       *http.Client
	lastError    string
	extractStats map[string]int // Track extraction success/failure
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
		client: &http.Client{
			Timeout: 120 * time.Second, // 2 minute timeout for slow systems
		},
		extractStats: make(map[string]int),
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
	// Increase buffer size to handle large responses (default is 64KB)
	// Set to 10MB to handle very long responses
	const maxScanTokenSize = 10 * 1024 * 1024
	buf := make([]byte, maxScanTokenSize)
	scanner.Buffer(buf, maxScanTokenSize)

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
	prompt := fmt.Sprintf(`Extract ALL named entities from this conversation as a JSON array.

For EACH entity (location, character, item, etc.), create an object with:
- entity_name: The proper name
- entity_type: "location", "character", "item", etc.
- description: Complete description
- attributes: MUST be a JSON object (not a string), like {"key": "value", "key2": "value2"}
- keywords: Array of searchable terms

CRITICAL: "attributes" MUST be an object with key-value pairs, NOT a string.

Example:
[
  {
    "entity_name": "The Red Tavern",
    "entity_type": "location",
    "description": "A bustling tavern in the merchant district with a large fireplace",
    "attributes": {
      "atmosphere": "warm and crowded",
      "location": "merchant district",
      "features": "large fireplace, private rooms upstairs"
    },
    "keywords": ["tavern", "red", "merchant", "inn", "fireplace"]
  }
]

If there are no specific key-value attributes, use: "attributes": {}

Q: %s
A: %s

Return ONLY the JSON array, no explanation:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	response = strings.TrimSpace(response)

	// Try to extract JSON from response (handle markdown code blocks, extra text, etc.)
	jsonStr := extractJSON(response, true) // true = expect array
	if jsonStr == "" {
		c.lastError = fmt.Sprintf("ExtractEntitySheets: No JSON found in response: %s", response[:min(200, len(response))])
		c.extractStats["entity_sheets_failed"]++
		return nil, nil
	}

	var result []EntitySheetResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		// Store error for debugging
		c.lastError = fmt.Sprintf("ExtractEntitySheets JSON parse error: %v | JSON: %s", err, jsonStr[:min(200, len(jsonStr))])
		c.extractStats["entity_sheets_failed"]++
		return nil, nil
	}

	c.extractStats["entity_sheets_success"]++
	return result, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// extractJSON robustly extracts JSON from LLM responses
// Handles: markdown code blocks (```json), extra text, explanations
func extractJSON(response string, expectArray bool) string {
	response = strings.TrimSpace(response)

	// Remove markdown code blocks
	if strings.Contains(response, "```") {
		// Extract content between ```json and ``` or ``` and ```
		start := strings.Index(response, "```json")
		if start == -1 {
			start = strings.Index(response, "```")
		}
		if start != -1 {
			start = strings.Index(response[start:], "\n")
			if start != -1 {
				response = response[start+1:]
				end := strings.Index(response, "```")
				if end != -1 {
					response = response[:end]
				}
			}
		}
	}

	response = strings.TrimSpace(response)

	// Find JSON structure
	if expectArray {
		startIdx := strings.Index(response, "[")
		endIdx := strings.LastIndex(response, "]")
		if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
			return response[startIdx : endIdx+1]
		}
	} else {
		startIdx := strings.Index(response, "{")
		endIdx := strings.LastIndex(response, "}")
		if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
			return response[startIdx : endIdx+1]
		}
	}

	return ""
}

// ExtractStructuredQA extracts who/what/why/when/where/how structure
func (c *OllamaClient) ExtractStructuredQA(model, userMsg, assistantMsg string) (*StructuredQAResult, error) {
	prompt := fmt.Sprintf(`Extract key information from this Q&A using the 5W1H framework.

Fill in ALL applicable fields. For location descriptions, focus on spatial details in "where" and environmental details in "what".

Return ONLY valid JSON in this exact format:
{
  "who": "people/characters/entities involved or mentioned",
  "what": "what is described, happening, or exists",
  "why": "purpose, significance, or reason",
  "when": "time period, era, or temporal context",
  "where": "location, place, or spatial relationships",
  "how": "mechanism, structure, or process",
  "keywords": ["searchable", "terms", "from", "content"]
}

Use empty string "" for fields that don't apply.

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

	jsonStr := extractJSON(response, false) // false = expect object
	if jsonStr == "" {
		c.lastError = fmt.Sprintf("ExtractStructuredQA: No JSON found in response: %s", response[:min(200, len(response))])
		c.extractStats["structured_qa_failed"]++
		return nil, nil
	}

	var result StructuredQAResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		c.lastError = fmt.Sprintf("ExtractStructuredQA JSON parse error: %v | JSON: %s", err, jsonStr[:min(200, len(jsonStr))])
		c.extractStats["structured_qa_failed"]++
		return nil, nil
	}

	c.extractStats["structured_qa_success"]++
	return &result, nil
}

// ExtractKeyValuePairs extracts key-value mappings for entity registry
func (c *OllamaClient) ExtractKeyValuePairs(model, userMsg, assistantMsg string) ([]KeyValuePair, error) {
	prompt := fmt.Sprintf(`Extract entity registry entries as key-value pairs.

For EACH named thing (person, place, item, concept), create an entry:
- key: The proper name (e.g., "The Broken Tower", "Aria the Merchant")
- value: Complete description with ALL details mentioned
- keywords: Searchable terms including synonyms and related concepts

Examples:
Location: {"key": "The Whispering Woods", "value": "Dark forest north of town, known for strange sounds at night", "keywords": ["woods", "forest", "whispering", "dark", "haunted"]}
Character: {"key": "Lord Vex", "value": "Cruel ruler of the northern provinces, wears black armor", "keywords": ["vex", "lord", "ruler", "northern", "armor", "cruel"]}

Return ONLY a JSON array with ALL entities found:

Q: %s
A: %s

JSON array:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	jsonStr := extractJSON(response, true) // true = expect array
	if jsonStr == "" {
		c.lastError = fmt.Sprintf("ExtractKeyValuePairs: No JSON found in response: %s", response[:min(200, len(response))])
		c.extractStats["kv_pairs_failed"]++
		return nil, nil
	}

	var result []KeyValuePair
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		c.lastError = fmt.Sprintf("ExtractKeyValuePairs JSON parse error: %v | JSON: %s", err, jsonStr[:min(200, len(jsonStr))])
		c.extractStats["kv_pairs_failed"]++
		return nil, nil
	}

	c.extractStats["kv_pairs_success"]++
	return result, nil
}

// CanonicalQA represents canonical question-answer pairs extracted from content
type CanonicalQA struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// ExtractCanonicalQA extracts canonical Q&A pairs from conversation
// Example: "who is the beggar" becomes "Who is The Beggar of Somewhere?"
func (c *OllamaClient) ExtractCanonicalQA(model, userMsg, assistantMsg string) ([]CanonicalQA, error) {
	prompt := fmt.Sprintf(`Create canonical question-answer pairs from this conversation.

For EACH fact, entity, or concept mentioned, create a well-formed Q&A:
- Use proper capitalization and punctuation
- Questions should be complete and specific
- Answers should be concise but complete
- Include variations: "What is X?", "Where is X?", "Who is X?", "What does X do?"

Examples:
User asks: "tell me about the tower"
→ [{"question": "What is the tower?", "answer": "An ancient stone tower on the hill"}]

User asks: "where is the market"
→ [{"question": "Where is the market?", "answer": "In the center of town, near the fountain"}]

Return ONLY a JSON array. Extract 2-5 Q&A pairs covering all key information:

Q: %s
A: %s

JSON array:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	jsonStr := extractJSON(response, true) // true = expect array
	if jsonStr == "" {
		c.lastError = fmt.Sprintf("ExtractCanonicalQA: No JSON found in response: %s", response[:min(200, len(response))])
		c.extractStats["canonical_qa_failed"]++
		return nil, nil
	}

	var result []CanonicalQA
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		c.lastError = fmt.Sprintf("ExtractCanonicalQA JSON parse error: %v | JSON: %s", err, jsonStr[:min(200, len(jsonStr))])
		c.extractStats["canonical_qa_failed"]++
		return nil, nil
	}

	c.extractStats["canonical_qa_success"]++
	return result, nil
}

// QueryEnhancement represents an enhanced query with extracted entities
type QueryEnhancement struct {
	OriginalQuery      string   `json:"original_query"`
	EnhancedQueries    []string `json:"enhanced_queries"`
	ExtractedEntities  []string `json:"extracted_entities"`
	CanonicalForm      string   `json:"canonical_form"`
}

// EnhanceQuery extracts entities and reformulates queries for better matching
func (c *OllamaClient) EnhanceQuery(model, query string) (*QueryEnhancement, error) {
	prompt := fmt.Sprintf(`Analyze this query and enhance it for semantic search.
Extract entities, create canonical form, and generate alternative phrasings.

Return ONLY a JSON object with:
- original_query: the input query
- enhanced_queries: array of alternative phrasings (3-5 variations)
- extracted_entities: array of key entities/concepts
- canonical_form: well-formed question with proper capitalization

Query: %s

JSON:`, query)

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

	var result QueryEnhancement
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, nil
	}

	return &result, nil
}

// QuestionKeyPair represents a generated question mapped to content
type QuestionKeyPair struct {
	Question string `json:"question"`
	Keywords []string `json:"keywords"`
}

// ExtractQuestionKeys generates questions that would lead to this content
func (c *OllamaClient) ExtractQuestionKeys(model, userMsg, assistantMsg string) ([]QuestionKeyPair, error) {
	prompt := fmt.Sprintf(`Read this conversation and generate questions that someone might ask to retrieve this information.

Think: "If someone wanted to find this content, what would they ask?"

Create 3-7 diverse questions covering:
- Direct questions about the main topic
- Questions about specific details mentioned
- Questions using different phrasings
- Questions from different perspectives

Return ONLY a JSON array:
[
  {
    "question": "What is the Tower of Nothingness?",
    "keywords": ["tower", "nothingness", "location"]
  },
  {
    "question": "Where can I find the Tower of Nothingness?",
    "keywords": ["tower", "location", "find"]
  }
]

User asked: %s
Assistant answered: %s

JSON array:`, userMsg, assistantMsg)

	chatMessages := []ChatMessage{
		{Role: "user", Content: prompt},
	}

	response, err := c.Chat(model, chatMessages)
	if err != nil {
		return nil, err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		c.lastError = fmt.Sprintf("ExtractQuestionKeys: No JSON found in response: %s", response[:min(200, len(response))])
		c.extractStats["question_keys_failed"]++
		return nil, nil
	}

	var result []QuestionKeyPair
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		c.lastError = fmt.Sprintf("ExtractQuestionKeys JSON parse error: %v | JSON: %s", err, jsonStr[:min(200, len(jsonStr))])
		c.extractStats["question_keys_failed"]++
		return nil, nil
	}

	c.extractStats["question_keys_success"]++
	return result, nil
}

// GetExtractionStats returns statistics about LLM extraction success/failure
func (c *OllamaClient) GetExtractionStats() map[string]int {
	return c.extractStats
}

// GetLastError returns the last extraction error for debugging
func (c *OllamaClient) GetLastError() string {
	return c.lastError
}

// ResetExtractionStats clears the extraction statistics
func (c *OllamaClient) ResetExtractionStats() {
	c.extractStats = make(map[string]int)
	c.lastError = ""
}
