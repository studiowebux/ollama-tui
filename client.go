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
