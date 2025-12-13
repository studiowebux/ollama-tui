package main

import (
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type switchProjectMsg struct {
	projectID string
}

// sortChunksByTime sorts chunks by CreatedAt timestamp (newest first)
func sortChunksByTime(chunks []VectorChunk) {
	sort.Slice(chunks, func(i, j int) bool {
		return chunks[i].CreatedAt.After(chunks[j].CreatedAt)
	})
}

// Project Switcher View
func (m model) renderProjectSwitcherView() string {
	title := titleStyle.Render("Project Switcher")
	help := helpStyle.Render("↑/↓: navigate | enter: switch | n: new project | d: delete | esc: back")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	if len(m.projects) == 0 {
		content.WriteString(helpStyle.Render("No projects found.") + "\n")
	} else {
		for i, project := range m.projects {
			cursor := " "
			if i == m.projectCursor {
				cursor = ">"
			}

			currentMarker := ""
			if project.ID == m.config.CurrentProject {
				currentMarker = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(" [CURRENT]")
			}

			projectLine := fmt.Sprintf("%s %s%s", cursor, project.Name, currentMarker)
			if i == m.projectCursor {
				projectLine = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render(projectLine)
			}
			content.WriteString(projectLine + "\n")
		}
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleProjectSwitcherViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		m.currentView = chatListView
		return m, m.loadChats

	case "up", "k":
		if m.projectCursor > 0 {
			m.projectCursor--
		}

	case "down", "j":
		if m.projectCursor < len(m.projects)-1 {
			m.projectCursor++
		}

	case "enter":
		if len(m.projects) > 0 {
			selectedProject := m.projects[m.projectCursor]
			return m, m.switchProject(selectedProject.ID)
		}

	case "n":
		// Create new project (simplified for now)
		newProject := &Project{
			Name: fmt.Sprintf("Project %d", len(m.projects)+1),
		}
		if err := m.projectManager.CreateProject(newProject); err == nil {
			m.projects = m.projectManager.ListProjects()
		}

	case "d":
		if len(m.projects) > 0 && m.projectCursor < len(m.projects) {
			selectedProject := m.projects[m.projectCursor]
			if selectedProject.ID != "default" {
				m.projectManager.DeleteProject(selectedProject.ID)
				m.projects = m.projectManager.ListProjects()
				if m.projectCursor >= len(m.projects) {
					m.projectCursor = len(m.projects) - 1
				}
			}
		}
	}

	return m, nil
}

func (m *model) switchProject(projectID string) tea.Cmd {
	return func() tea.Msg {
		// Update config
		m.config.CurrentProject = projectID
		if err := m.config.Save(); err != nil {
			return errMsg{err: err}
		}

		// Switch storage
		if err := m.storage.SwitchProject(projectID); err != nil {
			return errMsg{err: err}
		}

		// Switch vector DB
		if err := m.vectorDB.SwitchProject(projectID); err != nil {
			return errMsg{err: err}
		}

		return switchProjectMsg{projectID: projectID}
	}
}

// Knowledge Base View
func (m model) renderKnowledgeBaseView() string {
	title := titleStyle.Render("Knowledge Base - All Vector Chunks")
	help := helpStyle.Render("↑/↓: navigate | enter: view details | v: mark verified | b: mark bad | d: delete | esc: back")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	content.WriteString(helpStyle.Render(fmt.Sprintf("Total chunks: %d", len(m.kbChunks))) + "\n\n")

	if len(m.kbChunks) == 0 {
		content.WriteString(helpStyle.Render("No chunks found. Start a conversation and press Ctrl+B to vectorize.") + "\n")
	} else {
		// Show chunks with strategy badges
		displayStart := m.kbCursor - 5
		if displayStart < 0 {
			displayStart = 0
		}
		displayEnd := displayStart + 15
		if displayEnd > len(m.kbChunks) {
			displayEnd = len(m.kbChunks)
		}

		for i := displayStart; i < displayEnd; i++ {
			chunk := m.kbChunks[i]
			cursor := " "
			if i == m.kbCursor {
				cursor = ">"
			}

			// Strategy badge
			strategyBadge := getStrategyBadge(chunk.Strategy)

			// Status indicators
			status := ""
			if chunk.Metadata.MarkedBad {
				status = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Render("[BAD]")
			} else if chunk.Metadata.Verified {
				status = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("[OK]")
			}

			// Timestamp
			timestamp := chunk.CreatedAt.Format("15:04:05")
			timestampStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("243"))

			// Content preview
			preview := truncateString(chunk.Content, 45)

			chunkLine := fmt.Sprintf("%s %s %s %s %s", cursor, timestampStyle.Render(timestamp), strategyBadge, preview, status)
			if i == m.kbCursor {
				chunkLine = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render(chunkLine)
			}
			content.WriteString(chunkLine + "\n")
		}

		content.WriteString(helpStyle.Render(fmt.Sprintf("\nShowing %d-%d of %d", displayStart+1, displayEnd, len(m.kbChunks))) + "\n")
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleKnowledgeBaseViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		m.currentView = chatListView
		return m, m.loadChats

	case "up", "k":
		if m.kbCursor > 0 {
			m.kbCursor--
		}

	case "down", "j":
		if m.kbCursor < len(m.kbChunks)-1 {
			m.kbCursor++
		}

	case "enter":
		if len(m.kbChunks) > 0 {
			m.selectedChunk = &m.kbChunks[m.kbCursor]
			m.currentView = chunkDetailView
		}

	case "v":
		// Mark as verified
		if len(m.kbChunks) > 0 {
			chunk := &m.kbChunks[m.kbCursor]
			chunk.Metadata.Verified = true
			chunk.Metadata.MarkedBad = false
			// Save updated chunk (would need to add SaveChunk method)
		}

	case "b":
		// Mark as bad
		if len(m.kbChunks) > 0 {
			chunk := &m.kbChunks[m.kbCursor]
			m.vectorDB.MarkChunkBad(chunk.ID)
			chunk.Metadata.MarkedBad = true
			chunk.Metadata.Verified = false
		}

	case "d":
		// Delete chunk
		if len(m.kbChunks) > 0 {
			chunk := &m.kbChunks[m.kbCursor]
			m.vectorDB.DeleteChunk(chunk.ID)
			// Reload chunks
			m.kbChunks = m.vectorDB.GetAllChunks()
			sortChunksByTime(m.kbChunks)
			if m.kbCursor >= len(m.kbChunks) {
				m.kbCursor = len(m.kbChunks) - 1
			}
		}
	}

	return m, nil
}

// Chunk Detail View
func (m model) renderChunkDetailView() string {
	title := titleStyle.Render("Chunk Details")
	help := helpStyle.Render("r: refine with LLM | b: mark bad | v: mark verified | d: delete | esc: back")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	if m.selectedChunk == nil {
		content.WriteString("No chunk selected.\n")
	} else {
		chunk := m.selectedChunk

		// Strategy
		content.WriteString(helpStyle.Render("Strategy: ") + getStrategyBadge(chunk.Strategy) + "\n")
		content.WriteString(helpStyle.Render("ID: ") + chunk.ID[:8] + "...\n")
		content.WriteString(helpStyle.Render("Content Type: ") + string(chunk.ContentType) + "\n\n")

		// Status
		status := "Normal"
		if chunk.Metadata.MarkedBad {
			status = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Render("Marked Bad")
		} else if chunk.Metadata.Verified {
			status = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("Verified")
		}
		content.WriteString(helpStyle.Render("Status: ") + status + "\n\n")

		// Content
		content.WriteString(helpStyle.Render("Content:") + "\n")
		content.WriteString(chunk.Content + "\n\n")

		// Canonical Q&A
		if len(chunk.CanonicalQuestions) > 0 {
			content.WriteString(helpStyle.Render("\nCanonical Questions:") + "\n")
			for i, q := range chunk.CanonicalQuestions {
				if i >= 3 {
					content.WriteString(helpStyle.Render(fmt.Sprintf("  ... and %d more", len(chunk.CanonicalQuestions)-3)) + "\n")
					break
				}
				content.WriteString("  • " + q + "\n")
			}
		}
		if chunk.CanonicalAnswer != "" {
			content.WriteString(helpStyle.Render("Canonical Answer: ") + chunk.CanonicalAnswer + "\n")
		}

		// Metadata
		if chunk.Metadata.EntityKey != "" {
			content.WriteString(helpStyle.Render("\nEntity Key: ") + chunk.Metadata.EntityKey + "\n")
		}
		if len(chunk.Metadata.SearchKeywords) > 0 {
			content.WriteString(helpStyle.Render("Keywords: ") + strings.Join(chunk.Metadata.SearchKeywords, ", ") + "\n")
		}
		if len(chunk.Metadata.CharacterRefs) > 0 {
			content.WriteString(helpStyle.Render("Characters: ") + strings.Join(chunk.Metadata.CharacterRefs, ", ") + "\n")
		}
		if len(chunk.Metadata.LocationRefs) > 0 {
			content.WriteString(helpStyle.Render("Locations: ") + strings.Join(chunk.Metadata.LocationRefs, ", ") + "\n")
		}

		// 5W1H structure
		hasWho := chunk.Metadata.Who != ""
		hasWhat := chunk.Metadata.What != ""
		hasWhere := chunk.Metadata.Where != ""
		if hasWho || hasWhat || hasWhere {
			content.WriteString(helpStyle.Render("\nStructured Info:") + "\n")
			if hasWho {
				content.WriteString("  Who: " + chunk.Metadata.Who + "\n")
			}
			if hasWhat {
				content.WriteString("  What: " + chunk.Metadata.What + "\n")
			}
			if hasWhere {
				content.WriteString("  Where: " + chunk.Metadata.Where + "\n")
			}
		}

		// Related chunks
		if len(chunk.Metadata.RelatedChunkIDs) > 0 {
			content.WriteString(helpStyle.Render(fmt.Sprintf("\nRelated Chunks: %d", len(chunk.Metadata.RelatedChunkIDs))) + "\n")
		}
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleChunkDetailViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		m.currentView = knowledgeBaseView
		return m, nil

	case "r":
		// Start refinement flow
		if m.selectedChunk != nil {
			m.originalChunk = &VectorChunk{
				ID:          m.selectedChunk.ID,
				ChatID:      m.selectedChunk.ChatID,
				Content:     m.selectedChunk.Content,
				ContentType: m.selectedChunk.ContentType,
				Strategy:    m.selectedChunk.Strategy,
				Embedding:   m.selectedChunk.Embedding,
				Metadata:    m.selectedChunk.Metadata,
				CreatedAt:   m.selectedChunk.CreatedAt,
			}
			m.refineMessages = []string{
				fmt.Sprintf("Current chunk content:\n\n%s\n\nHow would you like to improve this?", m.selectedChunk.Content),
			}
			m.refineRoles = []string{"assistant"}
			m.refinedContent = ""
			m.currentView = refineChunkView
			m.textarea.Focus()
			m.textarea.SetValue("")
		}
		return m, nil

	case "v":
		if m.selectedChunk != nil {
			m.selectedChunk.Metadata.Verified = true
			m.selectedChunk.Metadata.MarkedBad = false
			// Save would happen here
		}

	case "b":
		if m.selectedChunk != nil {
			m.vectorDB.MarkChunkBad(m.selectedChunk.ID)
			m.selectedChunk.Metadata.MarkedBad = true
		}

	case "d":
		if m.selectedChunk != nil {
			m.vectorDB.DeleteChunk(m.selectedChunk.ID)
			m.kbChunks = m.vectorDB.GetAllChunks()
			sortChunksByTime(m.kbChunks)
			m.currentView = knowledgeBaseView
		}
	}

	return m, nil
}

func getStrategyBadge(strategy ChunkStrategy) string {
	badgeColors := map[ChunkStrategy]string{
		StrategyFullQA:       "99",  // Purple
		StrategySentence:     "208", // Orange
		StrategyKeyValue:     "86",  // Green
		StrategyWhoWhatWhy:   "33",  // Blue
		StrategyEntitySheet:  "205", // Pink
		StrategyKeyword:      "214", // Yellow
		StrategyQuestionKey:  "51",  // Cyan
	}

	badgeNames := map[ChunkStrategy]string{
		StrategyFullQA:       "FULL",
		StrategySentence:     "SENT",
		StrategyKeyValue:     "K:V",
		StrategyWhoWhatWhy:   "5W1H",
		StrategyEntitySheet:  "ENT",
		StrategyKeyword:      "KEY",
		StrategyQuestionKey:  "Q=>A",
	}

	color, ok := badgeColors[strategy]
	if !ok {
		color = "241"
	}
	name, ok := badgeNames[strategy]
	if !ok {
		name = "UNK"
	}

	return lipgloss.NewStyle().Foreground(lipgloss.Color(color)).Render(fmt.Sprintf("[%s]", name))
}

// Refinement Chat View
func (m model) renderRefineChunkView() string {
	title := titleStyle.Render("Refine Chunk - Chat with LLM")
	help := helpStyle.Render("esc: cancel | enter: send message | ctrl+d: generate improved version")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	// Show conversation
	for i, msg := range m.refineMessages {
		role := m.refineRoles[i]
		var roleStyle lipgloss.Style
		var roleLabel string
		if role == "user" {
			roleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Bold(true)
			roleLabel = "You"
		} else {
			roleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true)
			roleLabel = "Assistant"
		}
		content.WriteString(roleStyle.Render(roleLabel+": ") + msg + "\n\n")
	}

	if m.streaming {
		content.WriteString(helpStyle.Render("Streaming...") + "\n\n")
	}

	content.WriteString(m.textarea.View() + "\n\n")
	content.WriteString(help)

	return content.String()
}

func (m *model) handleRefineChunkViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if msg.Type == tea.KeyEsc {
		m.currentView = chunkDetailView
		m.textarea.Blur()
		return m, nil
	}

	if msg.Type == tea.KeyCtrlD {
		// Generate final improved version
		return m, m.generateRefinedChunk()
	}

	if msg.Type == tea.KeyEnter {
		if m.streaming {
			return m, nil
		}
		userInput := strings.TrimSpace(m.textarea.Value())
		if userInput == "" {
			return m, nil
		}

		m.refineMessages = append(m.refineMessages, userInput)
		m.refineRoles = append(m.refineRoles, "user")
		m.textarea.SetValue("")

		return m, m.sendRefineMessage(userInput)
	}

	var cmd tea.Cmd
	m.textarea, cmd = m.textarea.Update(msg)
	return m, cmd
}

func (m *model) sendRefineMessage(userMsg string) tea.Cmd {
	return func() tea.Msg {
		// Build conversation history
		chatMessages := []ChatMessage{}
		for i, msg := range m.refineMessages {
			chatMessages = append(chatMessages, ChatMessage{
				Role:    m.refineRoles[i],
				Content: msg,
			})
		}

		// Get response from LLM
		response, err := m.client.Chat(m.config.Model, chatMessages)
		if err != nil {
			return errMsg{err: err}
		}

		return refineResponseMsg{response: response}
	}
}

type refineResponseMsg struct {
	response string
}

type refineGenerateMsg struct {
	content string
}

func (m *model) generateRefinedChunk() tea.Cmd {
	return func() tea.Msg {
		// Build conversation and ask for final improved version
		chatMessages := []ChatMessage{}
		for i, msg := range m.refineMessages {
			chatMessages = append(chatMessages, ChatMessage{
				Role:    m.refineRoles[i],
				Content: msg,
			})
		}

		// Add final prompt
		chatMessages = append(chatMessages, ChatMessage{
			Role:    "user",
			Content: "Based on our discussion, please provide the final improved version of the chunk content. Return ONLY the improved content, no explanations or markdown formatting.",
		})

		response, err := m.client.Chat(m.config.Model, chatMessages)
		if err != nil {
			return errMsg{err: err}
		}

		return refineGenerateMsg{content: strings.TrimSpace(response)}
	}
}

// Refinement Diff View
func (m model) renderRefineDiffView() string {
	title := titleStyle.Render("Review Changes")
	help := helpStyle.Render("a: accept (replace) | k: keep both | c: cancel | e: continue editing")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	// Show original
	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true).Render("ORIGINAL:") + "\n")
	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("241")).Render(m.originalChunk.Content) + "\n\n")

	// Show refined
	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Bold(true).Render("REFINED:") + "\n")
	content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("255")).Render(m.refinedContent) + "\n\n")

	// Show diff summary
	content.WriteString(helpStyle.Render("Options:") + "\n")
	content.WriteString("  a: Accept - Replace original chunk with refined version\n")
	content.WriteString("  k: Keep Both - Create new chunk, keep original\n")
	content.WriteString("  c: Cancel - Discard changes\n")
	content.WriteString("  e: Continue Editing - Go back to chat\n\n")

	content.WriteString(help)

	return content.String()
}

func (m *model) handleRefineDiffViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "a", "A":
		// Accept: Replace original chunk
		if m.originalChunk != nil && m.refinedContent != "" {
			m.selectedChunk.Content = m.refinedContent
			m.selectedChunk.Metadata.Verified = true

			// Re-generate embedding for updated content
			if embedding, err := m.client.GenerateEmbedding(m.config.VectorModel, m.refinedContent); err == nil {
				m.selectedChunk.Embedding = embedding
			}

			// Delete old chunk and add updated one
			m.vectorDB.DeleteChunk(m.originalChunk.ID)
			m.vectorDB.AddChunk(*m.selectedChunk)

			// Reload chunks
			m.kbChunks = m.vectorDB.GetAllChunks()
			sortChunksByTime(m.kbChunks)
		}
		m.currentView = knowledgeBaseView
		return m, nil

	case "k", "K":
		// Keep both: Create new chunk
		if m.refinedContent != "" {
			newChunk := VectorChunk{
				ChatID:      m.originalChunk.ChatID,
				Content:     m.refinedContent,
				ContentType: m.originalChunk.ContentType,
				Strategy:    m.originalChunk.Strategy,
				Metadata:    m.originalChunk.Metadata,
			}
			newChunk.Metadata.Verified = true
			newChunk.Metadata.ParentChunkID = m.originalChunk.ID

			// Generate embedding
			if embedding, err := m.client.GenerateEmbedding(m.config.VectorModel, m.refinedContent); err == nil {
				newChunk.Embedding = embedding
				m.vectorDB.AddChunk(newChunk)
			}

			// Reload chunks
			m.kbChunks = m.vectorDB.GetAllChunks()
			sortChunksByTime(m.kbChunks)
		}
		m.currentView = knowledgeBaseView
		return m, nil

	case "c", "C":
		// Cancel: Discard changes
		m.currentView = chunkDetailView
		return m, nil

	case "e", "E":
		// Continue editing
		m.currentView = refineChunkView
		m.textarea.Focus()
		return m, nil
	}

	return m, nil
}
