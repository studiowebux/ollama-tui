package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

type Project struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type ProjectManager struct {
	projectsDir string
	projects    []*Project
}

func NewProjectManager() (*ProjectManager, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	projectsDir := filepath.Join(home, ".ollama-ui", "projects")
	if err := os.MkdirAll(projectsDir, 0755); err != nil {
		return nil, err
	}

	pm := &ProjectManager{
		projectsDir: projectsDir,
		projects:    []*Project{},
	}

	if err := pm.loadProjects(); err != nil {
		return nil, err
	}

	// Ensure default project exists
	if len(pm.projects) == 0 {
		defaultProject := &Project{
			ID:        "default",
			Name:      "Default Project",
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		if err := pm.CreateProject(defaultProject); err != nil {
			return nil, err
		}
	}

	return pm, nil
}

func (pm *ProjectManager) CreateProject(project *Project) error {
	if project.ID == "" {
		project.ID = uuid.New().String()
	}
	project.CreatedAt = time.Now()
	project.UpdatedAt = time.Now()

	// Create project directories
	projectPath := filepath.Join(pm.projectsDir, project.ID)
	chatsPath := filepath.Join(projectPath, "chats")
	vectorsPath := filepath.Join(projectPath, "vectors")

	if err := os.MkdirAll(chatsPath, 0755); err != nil {
		return err
	}
	if err := os.MkdirAll(vectorsPath, 0755); err != nil {
		return err
	}

	// Save project metadata
	data, err := json.MarshalIndent(project, "", "  ")
	if err != nil {
		return err
	}

	metaPath := filepath.Join(projectPath, "project.json")
	if err := os.WriteFile(metaPath, data, 0644); err != nil {
		return err
	}

	pm.projects = append(pm.projects, project)
	return nil
}

func (pm *ProjectManager) loadProjects() error {
	entries, err := os.ReadDir(pm.projectsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		metaPath := filepath.Join(pm.projectsDir, entry.Name(), "project.json")
		data, err := os.ReadFile(metaPath)
		if err != nil {
			continue
		}

		var project Project
		if err := json.Unmarshal(data, &project); err != nil {
			continue
		}

		pm.projects = append(pm.projects, &project)
	}

	return nil
}

func (pm *ProjectManager) ListProjects() []*Project {
	return pm.projects
}

func (pm *ProjectManager) GetProject(id string) *Project {
	for _, project := range pm.projects {
		if project.ID == id {
			return project
		}
	}
	return nil
}

func (pm *ProjectManager) DeleteProject(id string) error {
	if id == "default" {
		return nil // Cannot delete default project
	}

	projectPath := filepath.Join(pm.projectsDir, id)
	if err := os.RemoveAll(projectPath); err != nil {
		return err
	}

	// Remove from list
	filtered := make([]*Project, 0)
	for _, project := range pm.projects {
		if project.ID != id {
			filtered = append(filtered, project)
		}
	}
	pm.projects = filtered

	return nil
}

func (pm *ProjectManager) GetProjectPath(projectID string) string {
	return filepath.Join(pm.projectsDir, projectID)
}

func (pm *ProjectManager) GetChatsPath(projectID string) string {
	return filepath.Join(pm.projectsDir, projectID, "chats")
}

func (pm *ProjectManager) GetVectorsPath(projectID string) string {
	return filepath.Join(pm.projectsDir, projectID, "vectors")
}
