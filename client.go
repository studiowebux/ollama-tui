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
