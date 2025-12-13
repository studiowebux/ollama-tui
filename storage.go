package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
)

type Message struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type Chat struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Storage struct {
	dataDir        string
	projectManager *ProjectManager
	currentProject string
}

func NewStorage(pm *ProjectManager, projectID string) (*Storage, error) {
	dataDir := pm.GetChatsPath(projectID)
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	backupDir := filepath.Join(home, ".ollama-tui", "backups")
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return nil, err
	}

	return &Storage{
		dataDir:        dataDir,
		projectManager: pm,
		currentProject: projectID,
	}, nil
}

func (s *Storage) SwitchProject(projectID string) error {
	dataDir := s.projectManager.GetChatsPath(projectID)
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return err
	}
	s.dataDir = dataDir
	s.currentProject = projectID
	return nil
}

func (s *Storage) CreateChat(model string) (*Chat, error) {
	chat := &Chat{
		ID:        uuid.New().String(),
		Title:     "New Chat",
		Model:     model,
		Messages:  []Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.SaveChat(chat); err != nil {
		return nil, err
	}

	return chat, nil
}

func (s *Storage) SaveChat(chat *Chat) error {
	chat.UpdatedAt = time.Now()

	data, err := json.MarshalIndent(chat, "", "  ")
	if err != nil {
		return err
	}

	path := filepath.Join(s.dataDir, chat.ID+".json")
	return os.WriteFile(path, data, 0644)
}

func (s *Storage) LoadChat(id string) (*Chat, error) {
	path := filepath.Join(s.dataDir, id+".json")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var chat Chat
	if err := json.Unmarshal(data, &chat); err != nil {
		return nil, err
	}

	return &chat, nil
}

func (s *Storage) ListChats() ([]*Chat, error) {
	files, err := os.ReadDir(s.dataDir)
	if err != nil {
		return nil, err
	}

	var chats []*Chat
	for _, file := range files {
		if filepath.Ext(file.Name()) != ".json" {
			continue
		}

		id := file.Name()[:len(file.Name())-5]
		chat, err := s.LoadChat(id)
		if err != nil {
			continue
		}

		chats = append(chats, chat)
	}

	sort.Slice(chats, func(i, j int) bool {
		return chats[i].UpdatedAt.After(chats[j].UpdatedAt)
	})

	return chats, nil
}

func (s *Storage) DeleteChat(id string) error {
	path := filepath.Join(s.dataDir, id+".json")
	return os.Remove(path)
}

func (s *Storage) AddMessage(chat *Chat, role, content string) error {
	msg := Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	}

	chat.Messages = append(chat.Messages, msg)

	if len(chat.Messages) <= 2 && chat.Title == "New Chat" {
		// Strip newlines and normalize whitespace for title
		titleContent := strings.ReplaceAll(content, "\n", " ")
		titleContent = strings.ReplaceAll(titleContent, "\r", " ")
		titleContent = strings.Join(strings.Fields(titleContent), " ") // Collapse multiple spaces

		if len(titleContent) > 50 {
			chat.Title = titleContent[:50] + "..."
		} else {
			chat.Title = titleContent
		}
	}

	return s.SaveChat(chat)
}

func (s *Storage) BackupChat(chat *Chat) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	backupDir := filepath.Join(home, ".ollama-tui", "backups")
	timestamp := time.Now().Format("20060102_150405")
	filename := chat.ID + "_" + timestamp + ".json"
	path := filepath.Join(backupDir, filename)

	data, err := json.MarshalIndent(chat, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// ClearAllChats deletes all chat files
func (s *Storage) ClearAllChats() error {
	files, err := os.ReadDir(s.dataDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			path := filepath.Join(s.dataDir, file.Name())
			if err := os.Remove(path); err != nil {
				return err
			}
		}
	}

	return nil
}
