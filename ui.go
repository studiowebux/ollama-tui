package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type view int

const (
	chatView view = iota
	chatListView
	settingsView
	vectorStatsView
	confirmResetView
	projectSwitcherView
	knowledgeBaseView
	chunkDetailView
	refineChunkView
	refineDiffView
	documentImportView
	strategySelectionView
)

type model struct {
	storage           *Storage
	client            *OllamaClient
	config            *Config
	vectorDB          *VectorDB
	ragEngine         *RAGEngine
	mlScorer          *MLScorer
	projectManager    *ProjectManager
	currentView       view
	currentChat       *Chat
	chats             []*Chat
	projects          []*Project
	projectCursor     int
	kbChunks          []VectorChunk
	kbCursor          int
	selectedChunk     *VectorChunk
	originalChunk     *VectorChunk
	refinedContent    string
	refineMessages    []string
	refineRoles       []string
	textarea          textarea.Model
	viewport          viewport.Model
	messages          []string
	messageRoles      []string
	streaming         bool
	summarizing       bool
	vectorizing       bool
	vectorProgress    string
	vectorProgressChan chan tea.Msg
	err               error
	width             int
	height            int
	chatListCursor    int
	settingsInput     string
	settingsFocus     int
	models            []string
	modelCursor       int
	chunkChan         chan string
	errChan           chan error
	endpointInput     textarea.Model
	editingEndpoint   bool
	summaryInput      textarea.Model
	editingSummary    bool
	contextSize       int
	lastKeyG          bool
	lastVectorResults []SearchResult
	vectorContextUsed bool
	lastVectorDebug   string

	// Document import
	docImporter        *DocumentImporter
	importPath         string
	importProgress     []string
	importing          bool
	importCancelled    bool
	scannedFiles       []string
	importCursor       int
	importProgressChan chan string
	importCancelChan   chan bool
	selectedStrategies map[string]bool
	strategyCursor     int
	importAll          bool // Track if importing all or single file

	// Vector stats view scroll
	vectorStatsScroll int

	// Chunk detail view scroll
	chunkDetailScroll int

	// Rating system
	pendingRating     bool // Waiting for user to rate the last message
	pendingRatingIndex int  // Index of message being rated

	// Refinement system
	refining          bool   // Currently refining an answer
	refinementStatus  string // Current refinement status message
	lastRAGResult     *RAGResult // Last RAG result for refinement
	lastUserQuery     string     // Last user query for refinement
}

type streamChunkMsg string
type streamDoneMsg struct{}
type streamStartMsg struct {
	chunkChan chan string
	errChan   chan error
}
type errMsg struct{ err error }
type contextSizeMsg int
type resetCompleteMsg struct{}
type vectorizeStepMsg struct{ step string }

var (
	titleStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("205"))

	helpStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("241"))

	userStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("86")).
			Bold(true)

	assistantStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("141")).
			Bold(true)

	errorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("196")).
			Bold(true)

	thinkingStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("243"))
)

func initialModel(storage *Storage, client *OllamaClient, config *Config, vectorDB *VectorDB, pm *ProjectManager) model {
	ta := textarea.New()
	ta.Placeholder = "Type your message..."
	ta.Focus()
	ta.CharLimit = 0
	ta.SetWidth(80)
	ta.SetHeight(3)

	endpointTa := textarea.New()
	endpointTa.Placeholder = "http://localhost:11434"
	endpointTa.CharLimit = 0
	endpointTa.SetWidth(60)
	endpointTa.SetHeight(1)
	endpointTa.SetValue(config.Endpoint)

	summaryTa := textarea.New()
	summaryTa.Placeholder = "Enter summary prompt..."
	summaryTa.CharLimit = 0
	summaryTa.SetWidth(60)
	summaryTa.SetHeight(3)
	summaryTa.SetValue(config.SummaryPrompt)

	vp := viewport.New(80, 20)
	vp.SetContent("")

	ragEngine := NewRAGEngine(client, vectorDB, config)

	// Initialize ML scorer if explicitly enabled in config
	var mlScorer *MLScorer
	if config.MLEnableScoring && config.MLModelPath != "" && config.MLMetadataPath != "" {
		var err error
		mlScorer, err = NewMLScorer(config.MLModelPath, config.MLMetadataPath, config.MLOnnxLibPath)
		if err != nil {
			// Log warning but continue with heuristic scoring
			fmt.Fprintf(os.Stderr, "Warning: Failed to load ML model, using heuristic scoring: %v\n", err)
			mlScorer = nil
		}
	}

	return model{
		storage:            storage,
		client:             client,
		config:             config,
		vectorDB:           vectorDB,
		ragEngine:          ragEngine,
		mlScorer:           mlScorer,
		projectManager:     pm,
		currentView:        chatListView,
		textarea:          ta,
		viewport:          vp,
		messages:          []string{},
		endpointInput:     endpointTa,
		summaryInput:      summaryTa,
		selectedStrategies: make(map[string]bool),
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(textarea.Blink, m.loadChats)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = msg.Width
		m.textarea.SetWidth(msg.Width)

		// Calculate viewport height properly
		// Title: 1 line
		// Blank line: 1 line
		// Viewport: variable
		// Blank line: 1 line
		// Textarea: 3 lines
		// Status: 0-1 lines (assume 1)
		// Help: 1-3 lines depending on wrapping (assume 2 for width < 150, else 1)
		usedHeight := 1 + 1 + 1 + 3 + 1 // title + blank + blank + textarea + status

		// Help text wrapping estimation
		helpTextLength := 150 // approximate length of help text
		if msg.Width > 0 {
			helpLines := (helpTextLength / msg.Width) + 1
			if helpLines > 3 {
				helpLines = 3 // cap at 3 lines
			}
			usedHeight += helpLines
		} else {
			usedHeight += 2 // default to 2 lines for help
		}

		m.viewport.Height = msg.Height - usedHeight
		if m.viewport.Height < 5 {
			m.viewport.Height = 5 // Absolute minimum
		}

	case tea.KeyMsg:
		switch m.currentView {
		case chatView:
			return m.handleChatViewKeys(msg)
		case chatListView:
			return m.handleChatListViewKeys(msg)
		case settingsView:
			return m.handleSettingsViewKeys(msg)
		case vectorStatsView:
			return m.handleVectorStatsViewKeys(msg)
		case projectSwitcherView:
			return m.handleProjectSwitcherViewKeys(msg)
		case knowledgeBaseView:
			return m.handleKnowledgeBaseViewKeys(msg)
		case chunkDetailView:
			return m.handleChunkDetailViewKeys(msg)
		case refineChunkView:
			return m.handleRefineChunkViewKeys(msg)
		case refineDiffView:
			return m.handleRefineDiffViewKeys(msg)
		case confirmResetView:
			return m.handleConfirmResetViewKeys(msg)
		case documentImportView:
			return m.handleDocumentImportViewKeys(msg)
		case strategySelectionView:
			return m.handleStrategySelectionViewKeys(msg)
		}

	case tea.MouseMsg:
		// Handle mouse wheel scrolling in chat view
		if m.currentView == chatView {
			if msg.Type == tea.MouseWheelUp {
				m.viewport.LineUp(3)
			} else if msg.Type == tea.MouseWheelDown {
				m.viewport.LineDown(3)
			}
		}
		// Don't return - let it fall through to viewport.Update

	case streamStartMsg:
		m.streaming = true
		m.chunkChan = msg.chunkChan
		m.errChan = msg.errChan
		return m, m.waitForChunks(m.chunkChan, m.errChan)

	case streamChunkMsg:
		if len(m.messages) > 0 {
			m.messages[len(m.messages)-1] += string(msg)
		}
		m.updateViewport()
		return m, m.waitForChunks(m.chunkChan, m.errChan)

	case streamDoneMsg:
		m.streaming = false
		if m.currentChat != nil && len(m.messages) > 0 {
			assistantMsg := m.messages[len(m.messages)-1]
			m.storage.AddMessage(m.currentChat, "assistant", assistantMsg)

			// Trigger refinement if enabled
			if m.config.EnableRefinement && m.lastRAGResult != nil {
				return m, m.refineAnswer(m.lastUserQuery, assistantMsg)
			}
		}
		return m, nil

	case refinementStartMsg:
		m.refining = true
		m.refinementStatus = "Analyzing answer quality..."
		return m, nil

	case refinementStatusMsg:
		m.refinementStatus = string(msg)
		return m, nil

	case refinementDoneMsg:
		m.refining = false
		m.refinementStatus = ""

		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}

		// Replace the last message with refined answer if it improved
		if msg.result.WasRefined && len(m.messages) > 0 {
			m.messages[len(m.messages)-1] = msg.result.FinalAnswer

			// Update in storage
			if m.currentChat != nil && len(m.currentChat.Messages) > 0 {
				m.currentChat.Messages[len(m.currentChat.Messages)-1].Content = msg.result.FinalAnswer
				m.storage.SaveChat(m.currentChat)
			}

			m.updateViewport()
		}
		return m, nil

	case errMsg:
		m.err = msg.err
		m.streaming = false
		return m, nil

	case []*Chat:
		m.chats = msg
		return m, nil

	case newChatMsg:
		m.currentChat = msg.chat
		m.messages = []string{}
		m.messageRoles = []string{}
		m.currentView = chatView
		m.viewport.SetContent("")
		m.viewport.GotoTop()
		m.textarea.Reset()
		return m, m.fetchContextSize

	case []string:
		m.models = msg
		if len(m.models) > 0 && m.config.Model == "" {
			m.config.Model = m.models[0]
			return m, m.fetchContextSize
		}
		return m, nil

	case vectorizeStartMsg:
		m.vectorizing = true
		m.vectorProgress = "Starting..."
		return m, m.doVectorize()

	case vectorizeStepMsg:
		m.vectorProgress = msg.step
		// Continue listening for more progress messages
		return m, m.waitForVectorProgress

	case vectorizeProgressMsg:
		m.vectorProgress = fmt.Sprintf("Pair %d/%d", msg.current, msg.total)
		// Continue listening for more progress messages
		return m, m.waitForVectorProgress

	case vectorizeMsg:
		m.vectorizing = false
		m.vectorProgress = ""
		if m.vectorProgressChan != nil {
			close(m.vectorProgressChan)
			m.vectorProgressChan = nil
		}
		return m, nil

	case scanCompleteMsg:
		m.scannedFiles = msg.files
		return m, nil

	case importProgressMsg:
		m.importProgress = append(m.importProgress, msg.message)
		// Keep only last 20 lines to avoid memory issues
		if len(m.importProgress) > 20 {
			m.importProgress = m.importProgress[len(m.importProgress)-20:]
		}
		// Continue listening for more progress (channel might still have messages)
		if m.importProgressChan != nil {
			return m, m.waitForImportProgress(m.importProgressChan)
		}
		return m, nil

	case importCompleteMsg:
		m.importing = false
		m.importCancelled = false
		m.importProgressChan = nil
		m.importCancelChan = nil
		// Keep the last progress message visible
		return m, nil

	case switchProjectMsg:
		m.currentView = chatListView
		return m, m.loadChats

	case refineResponseMsg:
		m.refineMessages = append(m.refineMessages, msg.response)
		m.refineRoles = append(m.refineRoles, "assistant")
		return m, nil

	case refineGenerateMsg:
		m.refinedContent = msg.content
		m.currentView = refineDiffView
		return m, nil

	case summarizeMsg:
		m.summarizing = false
		if m.currentChat != nil {
			m.currentChat.Messages = []Message{
				{
					Role:      "system",
					Content:   msg.summary,
					Timestamp: m.currentChat.CreatedAt,
				},
			}
		}
		m.messages = []string{msg.summary}
		m.messageRoles = []string{"system"}
		m.updateViewport()
		return m, nil

	case contextSizeMsg:
		m.contextSize = int(msg)
		return m, nil

	case resetCompleteMsg:
		m.chats = []*Chat{}
		m.currentChat = nil
		m.messages = []string{}
		m.messageRoles = []string{}
		m.lastVectorResults = nil
		m.lastVectorDebug = ""
		m.vectorContextUsed = false
		m.currentView = chatListView
		return m, m.loadChats
	}

	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

func (m model) View() string {
	switch m.currentView {
	case chatView:
		return m.renderChatView()
	case chatListView:
		return m.renderChatListView()
	case settingsView:
		return m.renderSettingsView()
	case vectorStatsView:
		return m.renderVectorStatsView()
	case confirmResetView:
		return m.renderConfirmResetView()
	case projectSwitcherView:
		return m.renderProjectSwitcherView()
	case knowledgeBaseView:
		return m.renderKnowledgeBaseView()
	case chunkDetailView:
		return m.renderChunkDetailView()
	case refineChunkView:
		return m.renderRefineChunkView()
	case refineDiffView:
		return m.renderRefineDiffView()
	case documentImportView:
		return m.renderDocumentImportView()
	case strategySelectionView:
		return m.renderStrategySelectionView()
	}
	return ""
}

func (m model) renderChatView() string {
	// Build title
	title := titleStyle.Render("Ollama Chat")
	if m.currentChat != nil {
		tokenCount := m.client.EstimateTokenCount(m.currentChat.Messages)
		title += " - " + m.currentChat.Title + " (" + m.config.Model + ")"

		if m.contextSize > 0 {
			tokenStyle := helpStyle
			if tokenCount > m.contextSize {
				tokenStyle = errorStyle
			} else if float64(tokenCount) > float64(m.contextSize)*0.8 {
				tokenStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
			}
			title += tokenStyle.Render(fmt.Sprintf(" [%d/%d tokens]", tokenCount, m.contextSize))
		} else {
			title += helpStyle.Render(fmt.Sprintf(" [~%d tokens]", tokenCount))
		}
	}

	// Build help text
	help := helpStyle.Render("esc: back | ctrl+j/k or pgup/pgdn: scroll | ctrl+r: rate | ctrl+n: new | ctrl+s: settings | ctrl+t: summarize | ctrl+b: vectorize | ctrl+v: vector info")

	// Build status
	status := ""
	if m.vectorContextUsed {
		vectorIndicator := lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("Vector context used")
		status = vectorIndicator + " "
	}
	if m.pendingRating {
		status += lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).Render("RATING MODE: Press 1-5 to rate | ESC to cancel")
	} else if m.vectorizing {
		if m.vectorProgress != "" {
			status += helpStyle.Render(fmt.Sprintf("Vectorizing conversation... %s", m.vectorProgress))
		} else {
			status += helpStyle.Render("Vectorizing conversation...")
		}
	} else if m.summarizing {
		status += helpStyle.Render("Summarizing conversation...")
	} else if m.refining {
		status += lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Render(fmt.Sprintf("Refining answer... %s", m.refinementStatus))
	} else if m.streaming {
		status += helpStyle.Render("Streaming...")
	}
	if m.err != nil {
		status = errorStyle.Render(fmt.Sprintf("Error: %v", m.err))
	}

	// Use lipgloss to properly layout components
	sections := []string{
		title,
		"", // blank line
		m.viewport.View(),
		"", // blank line
		m.textarea.View(),
	}

	if status != "" {
		sections = append(sections, status)
	}
	sections = append(sections, help)

	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

func (m model) renderChatListView() string {
	projectName := "default"
	if project := m.projectManager.GetProject(m.config.CurrentProject); project != nil {
		projectName = project.Name
	}
	title := titleStyle.Render(fmt.Sprintf("Chat History - Project: %s", projectName))
	modelInfo := helpStyle.Render(fmt.Sprintf("Current model: %s", m.config.Model))
	help := helpStyle.Render("↑/↓: navigate | enter: open | n: new chat | d: delete | p: projects | k: KB | i: import docs | s: settings | v: vector stats | r: reset all | q: quit")

	var content strings.Builder
	content.WriteString(title + " - " + modelInfo + "\n\n")

	if len(m.chats) == 0 {
		content.WriteString(helpStyle.Render("No chats yet. Press 'n' to create a new chat.") + "\n")
	} else {
		for i, chat := range m.chats {
			cursor := " "
			if i == m.chatListCursor {
				cursor = ">"
			}
			chatLine := fmt.Sprintf("%s %s (%s) - %d messages",
				cursor, chat.Title, chat.Model, len(chat.Messages))
			if i == m.chatListCursor {
				chatLine = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render(chatLine)
			}
			content.WriteString(chatLine + "\n")
		}
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m model) renderSettingsView() string {
	title := titleStyle.Render("Settings")
	help := helpStyle.Render("tab: next field | enter: save/edit | esc: cancel")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	// Endpoint field
	endpointLabel := "Endpoint:"
	if m.settingsFocus == 0 {
		endpointLabel = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("> " + endpointLabel)
	} else {
		endpointLabel = "  " + endpointLabel
	}
	content.WriteString(endpointLabel + "\n")

	if m.editingEndpoint && m.settingsFocus == 0 {
		content.WriteString("  " + m.endpointInput.View() + "\n")
	} else {
		endpointValue := "  " + m.config.Endpoint
		if m.settingsFocus == 0 {
			endpointValue = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(endpointValue)
		}
		content.WriteString(endpointValue + "\n")
	}
	content.WriteString("\n")

	// Model field
	modelLabel := "Model: " + m.config.Model
	if m.settingsFocus == 1 {
		modelLabel = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("> " + modelLabel)
	} else {
		modelLabel = "  " + modelLabel
	}
	content.WriteString(modelLabel + "\n")

	if len(m.models) > 0 && m.settingsFocus == 1 {
		content.WriteString("\nAvailable models:\n")
		for i, model := range m.models {
			cursor := " "
			if i == m.modelCursor {
				cursor = ">"
			}
			modelLine := fmt.Sprintf("%s %s", cursor, model)
			if i == m.modelCursor {
				modelLine = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(modelLine)
			}
			content.WriteString(modelLine + "\n")
		}
	}

	content.WriteString("\n")

	// Summary Prompt field
	summaryLabel := "Summary Prompt:"
	if m.settingsFocus == 2 {
		summaryLabel = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("> " + summaryLabel)
	} else {
		summaryLabel = "  " + summaryLabel
	}
	content.WriteString(summaryLabel + "\n")

	if m.editingSummary && m.settingsFocus == 2 {
		content.WriteString("  " + m.summaryInput.View() + "\n")
	} else {
		summaryPreview := m.config.SummaryPrompt
		if len(summaryPreview) > 60 {
			summaryPreview = summaryPreview[:60] + "..."
		}
		summaryValue := "  " + summaryPreview
		if m.settingsFocus == 2 {
			summaryValue = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(summaryValue)
		}
		content.WriteString(summaryValue + "\n")
	}

	content.WriteString("\n")

	// Vector Settings
	vectorLabel := "Vector DB:"
	if m.settingsFocus == 3 {
		vectorLabel = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render("> " + vectorLabel)
	} else {
		vectorLabel = "  " + vectorLabel
	}
	content.WriteString(vectorLabel + "\n")

	vectorStatus := "Disabled"
	if m.config.VectorEnabled {
		vectorStatus = fmt.Sprintf("Enabled (model: %s, topK: %d, threshold: %.2f)",
			m.config.VectorModel, m.config.VectorTopK, m.config.VectorSimilarity)
	}
	vectorValue := "  " + vectorStatus
	if m.settingsFocus == 3 {
		vectorValue = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(vectorValue)
	}
	content.WriteString(vectorValue + "\n")

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleChatViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Handle control keys first, before passing to textarea
	if msg.Type == tea.KeyCtrlC {
		return m, tea.Quit
	}
	if msg.Type == tea.KeyEsc {
		m.currentView = chatListView
		return m, m.loadChats
	}
	if msg.Type == tea.KeyCtrlN {
		return m, m.createNewChat
	}
	if msg.Type == tea.KeyCtrlL {
		m.currentView = chatListView
		return m, m.loadChats
	}
	if msg.Type == tea.KeyCtrlS {
		m.currentView = settingsView
		return m, m.loadModels
	}
	if msg.Type == tea.KeyCtrlT {
		return m, m.summarizeChat()
	}
	if msg.Type == tea.KeyCtrlB {
		return m, m.vectorizeChat()
	}
	if msg.Type == tea.KeyCtrlV {
		m.currentView = vectorStatsView
		return m, nil
	}

	// Handle scrolling with Ctrl+j/k and PgUp/PgDn
	if msg.Type == tea.KeyPgUp || msg.Type == tea.KeyCtrlK {
		m.viewport.LineUp(5)
		return m, nil
	}
	if msg.Type == tea.KeyPgDown || msg.Type == tea.KeyCtrlJ {
		m.viewport.LineDown(5)
		return m, nil
	}

	// Handle rating with Ctrl+R
	if msg.Type == tea.KeyCtrlR && !m.streaming && !m.pendingRating {
		// Find the last assistant message in currentChat.Messages (allow re-rating)
		if m.currentChat == nil {
			m.err = fmt.Errorf("no active chat")
			return m, nil
		}

		found := false
		for i := len(m.currentChat.Messages) - 1; i >= 0; i-- {
			if m.currentChat.Messages[i].Role == "assistant" {
				m.pendingRating = true
				m.pendingRatingIndex = i
				m.updateViewport()
				found = true
				break
			}
		}
		if !found {
			m.err = fmt.Errorf("no assistant messages to rate")
		}
		return m, nil
	}

	// Handle rating score keys (1-5)
	if m.pendingRating && (msg.String() >= "1" && msg.String() <= "5") {
		score := int(msg.String()[0] - '0')
		return m, m.rateMessage(m.pendingRatingIndex, score)
	}

	// Cancel rating with Esc
	if m.pendingRating && msg.Type == tea.KeyEsc {
		m.pendingRating = false
		m.updateViewport()
		return m, nil
	}

	// Handle enter for sending messages
	if msg.Type == tea.KeyEnter {
		if m.streaming || m.pendingRating {
			return m, nil
		}
		return m, m.sendMessage()
	}

	// Pass all other keys to textarea when not streaming and not rating
	if !m.streaming && !m.pendingRating {
		var cmd tea.Cmd
		m.textarea, cmd = m.textarea.Update(msg)
		return m, cmd
	}

	return m, nil
}

func (m *model) handleChatListViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "q", "ctrl+c":
		return m, tea.Quit

	case "up":
		if m.chatListCursor > 0 {
			m.chatListCursor--
		}

	case "down":
		if m.chatListCursor < len(m.chats)-1 {
			m.chatListCursor++
		}

	case "enter":
		if len(m.chats) > 0 {
			m.currentChat = m.chats[m.chatListCursor]
			m.messages = []string{}
			m.messageRoles = []string{}
			for _, msg := range m.currentChat.Messages {
				m.messages = append(m.messages, msg.Content)
				m.messageRoles = append(m.messageRoles, msg.Role)
			}
			m.updateViewport()
			m.currentView = chatView
			m.config.Model = m.currentChat.Model
			return m, m.fetchContextSize
		}

	case "n":
		return m, m.createNewChat

	case "d":
		if len(m.chats) > 0 {
			chat := m.chats[m.chatListCursor]
			m.storage.DeleteChat(chat.ID)
			if m.currentChat != nil && m.currentChat.ID == chat.ID {
				m.currentChat = nil
			}
			return m, m.loadChats
		}

	case "s":
		m.currentView = settingsView
		return m, m.loadModels

	case "v":
		m.currentView = vectorStatsView
		return m, nil

	case "p":
		m.projects = m.projectManager.ListProjects()
		m.projectCursor = 0
		m.currentView = projectSwitcherView
		return m, nil

	case "k":
		m.kbChunks = m.vectorDB.GetAllChunks()
		sortChunksByTime(m.kbChunks)
		m.kbCursor = 0
		m.currentView = knowledgeBaseView
		return m, nil

	case "i":
		m.importPath = "."
		m.scannedFiles = nil
		m.importCursor = 0
		m.currentView = documentImportView
		return m, m.scanDirectory()

	case "r":
		m.currentView = confirmResetView
		return m, nil
	}

	return m, nil
}

func (m *model) handleSettingsViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// If editing endpoint, handle textarea input
	if m.editingEndpoint && m.settingsFocus == 0 {
		switch msg.Type {
		case tea.KeyEnter:
			// Save endpoint
			newEndpoint := m.endpointInput.Value()
			if newEndpoint != "" {
				m.config.Endpoint = newEndpoint
				m.client.SetEndpoint(newEndpoint)
				m.config.Save()
			}
			m.editingEndpoint = false
			return m, m.loadModels
		case tea.KeyEsc:
			// Cancel editing
			m.editingEndpoint = false
			m.endpointInput.SetValue(m.config.Endpoint)
			return m, nil
		default:
			var cmd tea.Cmd
			m.endpointInput, cmd = m.endpointInput.Update(msg)
			return m, cmd
		}
	}

	// If editing summary prompt, handle textarea input
	if m.editingSummary && m.settingsFocus == 2 {
		switch msg.Type {
		case tea.KeyEnter:
			// Save summary prompt
			newPrompt := m.summaryInput.Value()
			if newPrompt != "" {
				m.config.SummaryPrompt = newPrompt
				m.config.Save()
			}
			m.editingSummary = false
			return m, nil
		case tea.KeyEsc:
			// Cancel editing
			m.editingSummary = false
			m.summaryInput.SetValue(m.config.SummaryPrompt)
			return m, nil
		default:
			var cmd tea.Cmd
			m.summaryInput, cmd = m.summaryInput.Update(msg)
			return m, cmd
		}
	}

	switch msg.String() {
	case "esc":
		m.editingEndpoint = false
		m.editingSummary = false
		m.currentView = chatListView
		return m, nil

	case "tab":
		m.editingEndpoint = false
		m.editingSummary = false
		m.settingsFocus = (m.settingsFocus + 1) % 4

	case "up", "k":
		if m.settingsFocus == 1 && len(m.models) > 0 {
			if m.modelCursor > 0 {
				m.modelCursor--
			}
		}

	case "down", "j":
		if m.settingsFocus == 1 && len(m.models) > 0 {
			if m.modelCursor < len(m.models)-1 {
				m.modelCursor++
			}
		}

	case "enter":
		if m.settingsFocus == 0 {
			// Start editing endpoint
			m.editingEndpoint = true
			m.endpointInput.Focus()
			return m, textarea.Blink
		} else if m.settingsFocus == 1 && len(m.models) > 0 {
			// Select model
			m.config.Model = m.models[m.modelCursor]
			m.config.Save()
			m.currentView = chatListView
			return m, m.fetchContextSize
		} else if m.settingsFocus == 2 {
			// Start editing summary prompt
			m.editingSummary = true
			m.summaryInput.Focus()
			return m, textarea.Blink
		} else if m.settingsFocus == 3 {
			// Toggle vector DB
			m.config.VectorEnabled = !m.config.VectorEnabled
			m.config.Save()
		}
	}

	return m, nil
}

func (m *model) updateViewport() {
	var content strings.Builder

	// Build display from m.currentChat.Messages (source of truth)
	// m.messages is kept in sync but may have an extra empty streaming message
	if m.currentChat != nil {
		for i := 0; i < len(m.currentChat.Messages); i++ {
			msg := &m.currentChat.Messages[i]

			if msg.Role == "user" {
				content.WriteString(userStyle.Render("You:") + "\n")
				content.WriteString(msg.Content + "\n\n")
			} else {
				content.WriteString(assistantStyle.Render("Assistant:") + "\n")
				content.WriteString(renderMessageWithThinking(msg.Content) + "\n")

				// Show rating for assistant messages
				if msg.Rating != nil {
					// Show existing rating with more visible styling
					ratingText := lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(fmt.Sprintf("  [Rated: %d/5]", msg.Rating.Score))
					content.WriteString(ratingText + "\n")
				} else if msg.Content != "" {
					// Show rating prompt for unrated messages
					if m.pendingRating && m.pendingRatingIndex == i {
						ratingPrompt := lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Bold(true).Render("  RATING MODE: Press 1-5 to rate | ESC to cancel")
						content.WriteString(ratingPrompt + "\n")
					} else if !m.streaming {
						// Only show hint when not streaming
						ratingHint := helpStyle.Render("  [Press Ctrl+R to rate]")
						content.WriteString(ratingHint + "\n")
					}
				}
				content.WriteString("\n")
			}
		}

		// If streaming, show the current incomplete assistant message
		if m.streaming && len(m.messages) > len(m.currentChat.Messages) {
			content.WriteString(assistantStyle.Render("Assistant:") + "\n")
			streamingMsg := m.messages[len(m.messages)-1]
			content.WriteString(renderMessageWithThinking(streamingMsg) + "\n")
			content.WriteString("\n")
		}

		// Add padding at bottom to ensure last lines are visible
		// Use half the viewport height to allow scrolling bottom content into view
		// without excessive blank space
		paddingLines := m.viewport.Height / 2
		if paddingLines < 3 {
			paddingLines = 3
		}
		content.WriteString(strings.Repeat("\n", paddingLines))
	}

	m.viewport.SetContent(content.String())
	// Only auto-scroll to bottom when streaming or if already at bottom
	// This allows users to scroll up without being forced back down
	if m.streaming {
		m.viewport.GotoBottom()
	}
}

func (m *model) sendMessage() tea.Cmd {
	userMsg := m.textarea.Value()
	if userMsg == "" {
		return nil
	}

	m.textarea.Reset()

	if m.currentChat == nil {
		return func() tea.Msg {
			return errMsg{err: fmt.Errorf("no active chat")}
		}
	}

	if err := m.storage.AddMessage(m.currentChat, "user", userMsg); err != nil {
		return func() tea.Msg {
			return errMsg{err: err}
		}
	}

	m.messages = append(m.messages, userMsg)
	m.messageRoles = append(m.messageRoles, "user")
	m.messages = append(m.messages, "")
	m.messageRoles = append(m.messageRoles, "assistant")
	m.streaming = true
	m.updateViewport()

	// Move context retrieval and message building to async function
	return func() tea.Msg {
		// Retrieve relevant context from vector DB
		relevantContext, err := m.retrieveRelevantContext(userMsg)
		if err != nil {
			return errMsg{err: fmt.Errorf("context retrieval failed: %w", err)}
		}

		// Build chat messages with proper context handling
		chatMessages := make([]ChatMessage, 0, len(m.currentChat.Messages)+1)

		// Add system instruction with context if available
		if relevantContext != "" {
			// Add instruction first
			chatMessages = append(chatMessages, ChatMessage{
				Role: "system",
				Content: `Answer questions using the provided context.

CRITICAL: If user specifies a word limit (e.g. "10 words max"), your answer MUST be that length or shorter. Do not write long explanations when brevity is requested.

Context:`,
			})

			// Add context as separate message
			chatMessages = append(chatMessages, ChatMessage{
				Role:    "system",
				Content: relevantContext,
			})
		}

		// Add conversation history (skip the last user message we already have)
		for i, msg := range m.currentChat.Messages {
			if i == len(m.currentChat.Messages)-1 && msg.Role == "user" {
				// Add final user message without context prepended
				chatMessages = append(chatMessages, ChatMessage{
					Role:    msg.Role,
					Content: msg.Content,
				})
			} else {
				chatMessages = append(chatMessages, ChatMessage{
					Role:    msg.Role,
					Content: msg.Content,
				})
			}
		}

		// Return streamStartMsg directly to begin streaming
		chunkChan := make(chan string, 100)
		errChan := make(chan error, 1)

		go func() {
			err := m.client.StreamChat(m.config.Model, chatMessages, func(chunk string) error {
				chunkChan <- chunk
				return nil
			})
			close(chunkChan)
			if err != nil {
				errChan <- err
			}
			close(errChan)
		}()

		return streamStartMsg{chunkChan: chunkChan, errChan: errChan}
	}
}

func (m model) streamResponse(messages []ChatMessage) tea.Cmd {
	return func() tea.Msg {
		chunkChan := make(chan string, 100)
		errChan := make(chan error, 1)

		go func() {
			err := m.client.StreamChat(m.config.Model, messages, func(chunk string) error {
				chunkChan <- chunk
				return nil
			})
			close(chunkChan)
			if err != nil {
				errChan <- err
			}
			close(errChan)
		}()

		return streamStartMsg{chunkChan: chunkChan, errChan: errChan}
	}
}

func (m model) waitForChunks(chunkChan chan string, errChan chan error) tea.Cmd {
	return func() tea.Msg {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				select {
				case err := <-errChan:
					if err != nil {
						return errMsg{err: err}
					}
				default:
				}
				return streamDoneMsg{}
			}
			return streamChunkMsg(chunk)
		case err := <-errChan:
			return errMsg{err: err}
		}
	}
}

func (m model) loadChats() tea.Msg {
	chats, err := m.storage.ListChats()
	if err != nil {
		return errMsg{err: err}
	}
	return chats
}

type newChatMsg struct {
	chat *Chat
}

func (m model) createNewChat() tea.Msg {
	chat, err := m.storage.CreateChat(m.config.Model)
	if err != nil {
		return errMsg{err: err}
	}
	return newChatMsg{chat: chat}
}

func (m model) loadModels() tea.Msg {
	models, err := m.client.ListModels()
	if err != nil {
		return errMsg{err: err}
	}
	return models
}

type summarizeMsg struct {
	summary string
}

type vectorizeMsg struct{}
type vectorizeStartMsg struct{}
type vectorizeProgressMsg struct {
	current int
	total   int
}

type refinementStartMsg struct{}
type refinementStatusMsg string
type refinementDoneMsg struct {
	result *RefinementResult
	err    error
}

func (m *model) rateMessage(messageIndex int, score int) tea.Cmd {
	if m.currentChat == nil || messageIndex >= len(m.currentChat.Messages) {
		return nil
	}

	msg := &m.currentChat.Messages[messageIndex]
	if msg.Role != "assistant" {
		return nil
	}

	// Find the corresponding user query
	userQuery := ""
	if messageIndex > 0 && m.currentChat.Messages[messageIndex-1].Role == "user" {
		userQuery = m.currentChat.Messages[messageIndex-1].Content
	}

	// Create rating
	msg.Rating = &Rating{
		Score:           score,
		Timestamp:       time.Now(),
		Query:           userQuery,
		ContextUsed:     m.vectorContextUsed,
		ContextChunks:   len(m.lastVectorResults),
		Model:           m.config.Model,
		VectorTopK:      m.config.VectorTopK,
		VectorSimilarity: m.config.VectorSimilarity,
	}

	// Save the chat
	if err := m.storage.SaveChat(m.currentChat); err != nil {
		m.err = fmt.Errorf("failed to save rating: %v", err)
		m.pendingRating = false
		m.updateViewport()
		return nil
	}

	// Clear rating state and update viewport
	m.pendingRating = false
	m.err = nil // Clear any previous errors
	m.updateViewport()
	// Go to bottom to show the newly added rating
	m.viewport.GotoBottom()

	// Return a no-op command to trigger re-render
	return func() tea.Msg {
		return nil
	}
}

func (m *model) refineAnswer(query, initialAnswer string) tea.Cmd {
	if m.ragEngine == nil || m.lastRAGResult == nil {
		return nil
	}

	return func() tea.Msg {
		// Send start message
		go func() {
			m.viewport.SetContent(m.viewport.View())
		}()

		progressChan := make(chan string, 10)
		defer close(progressChan)

		// Start background goroutine to send status messages
		go func() {
			for msg := range progressChan {
				// Send status updates to the UI
				go func(status string) {
					// This will be picked up by the Update function
				}(msg)
			}
		}()

		refinementEngine := NewRefinementEngine(m.client, m.ragEngine, m.config, m.mlScorer)
		result, err := refinementEngine.RefineAnswer(query, initialAnswer, m.lastRAGResult, m.config.Model, progressChan)

		return refinementDoneMsg{result: result, err: err}
	}
}

func (m *model) vectorizeChat() tea.Cmd {
	if m.currentChat == nil || len(m.currentChat.Messages) == 0 {
		return func() tea.Msg {
			return errMsg{err: fmt.Errorf("no chat to vectorize")}
		}
	}

	return func() tea.Msg {
		return vectorizeStartMsg{}
	}
}

func (m *model) doVectorize() tea.Cmd {
	cleanedMessages := make([]Message, len(m.currentChat.Messages))
	for i, msg := range m.currentChat.Messages {
		cleanedMessages[i] = Message{
			Role:      msg.Role,
			Content:   stripThinkingTags(msg.Content),
			Timestamp: msg.Timestamp,
		}
	}

	// Create progress channel and store it
	m.vectorProgressChan = make(chan tea.Msg, 100)

	// Start vectorization in background
	go func() {
		if err := m.vectorizeConversation(cleanedMessages, m.vectorProgressChan); err != nil {
			m.vectorProgressChan <- errMsg{err: fmt.Errorf("vectorization failed: %w", err)}
		} else {
			m.vectorProgressChan <- vectorizeMsg{}
		}
	}()

	// Return a command that waits for the next progress message
	return m.waitForVectorProgress
}

func (m *model) waitForVectorProgress() tea.Msg {
	if m.vectorProgressChan == nil {
		return nil
	}
	msg, ok := <-m.vectorProgressChan
	if !ok {
		return vectorizeMsg{}
	}
	return msg
}

func (m *model) summarizeChat() tea.Cmd {
	if m.currentChat == nil || len(m.currentChat.Messages) == 0 {
		return func() tea.Msg {
			return errMsg{err: fmt.Errorf("no chat to summarize")}
		}
	}

	m.summarizing = true

	return func() tea.Msg {
		if err := m.storage.BackupChat(m.currentChat); err != nil {
			return errMsg{err: fmt.Errorf("backup failed: %w", err)}
		}

		cleanedMessages := make([]Message, len(m.currentChat.Messages))
		for i, msg := range m.currentChat.Messages {
			cleanedMessages[i] = Message{
				Role:      msg.Role,
				Content:   stripThinkingTags(msg.Content),
				Timestamp: msg.Timestamp,
			}
		}

		// Vectorize conversation if enabled
		if m.config.VectorEnabled {
			if err := m.vectorizeConversation(cleanedMessages, nil); err != nil {
				return errMsg{err: fmt.Errorf("vectorization failed: %w", err)}
			}
		}

		summary, err := m.client.GenerateSummary(m.config.Model, m.config.SummaryPrompt, cleanedMessages)
		if err != nil {
			return errMsg{err: fmt.Errorf("summary generation failed: %w", err)}
		}

		m.currentChat.Messages = []Message{
			{
				Role:      "system",
				Content:   summary,
				Timestamp: m.currentChat.CreatedAt,
			},
		}

		if err := m.storage.SaveChat(m.currentChat); err != nil {
			return errMsg{err: fmt.Errorf("save failed: %w", err)}
		}

		return summarizeMsg{summary: summary}
	}
}

func stripThinkingTags(content string) string {
	thinkingPatterns := []string{
		"<think>", "</think>",
		"<thinking>", "</thinking>",
		"<thought>", "</thought>",
		"<internal>", "</internal>",
	}

	result := content
	start := -1
	depth := 0

	for i := 0; i < len(result); {
		found := false
		for _, pattern := range thinkingPatterns {
			if strings.HasPrefix(result[i:], pattern) {
				if strings.HasPrefix(pattern, "</") {
					depth--
					if depth == 0 && start != -1 {
						result = result[:start] + result[i+len(pattern):]
						i = start
						start = -1
						found = true
						break
					}
				} else {
					if depth == 0 {
						start = i
					}
					depth++
				}
				i += len(pattern)
				found = true
				break
			}
		}
		if !found {
			i++
		}
	}

	return strings.TrimSpace(result)
}

func renderMessageWithThinking(content string) string {
	thinkingPatterns := []struct {
		open  string
		close string
	}{
		{"<think>", "</think>"},
		{"<thinking>", "</thinking>"},
		{"<thought>", "</thought>"},
		{"<internal>", "</internal>"},
	}

	var result strings.Builder
	remaining := content

	for len(remaining) > 0 {
		foundThinking := false
		earliestPos := len(remaining)
		var matchedPattern struct {
			open  string
			close string
		}

		for _, pattern := range thinkingPatterns {
			if pos := strings.Index(remaining, pattern.open); pos != -1 && pos < earliestPos {
				earliestPos = pos
				matchedPattern = pattern
				foundThinking = true
			}
		}

		if !foundThinking {
			result.WriteString(remaining)
			break
		}

		result.WriteString(remaining[:earliestPos])

		closePos := strings.Index(remaining[earliestPos:], matchedPattern.close)
		if closePos == -1 {
			result.WriteString(remaining[earliestPos:])
			break
		}

		closePos += earliestPos
		thinkingContent := remaining[earliestPos : closePos+len(matchedPattern.close)]
		result.WriteString(thinkingStyle.Render(thinkingContent))

		remaining = remaining[closePos+len(matchedPattern.close):]
	}

	return result.String()
}

func (m model) fetchContextSize() tea.Msg {
	contextSize, err := m.client.GetContextSize(m.config.Model)
	if err != nil {
		return contextSizeMsg(4096)
	}
	return contextSizeMsg(contextSize)
}

// vectorizeConversation creates embeddings for message pairs using multiple strategies
func (m *model) vectorizeConversation(messages []Message, progressChan chan<- tea.Msg) error {
	var prevChunkID string

	// Count Q&A pairs
	totalPairs := 0
	for i := 0; i < len(messages)-1; i += 2 {
		if i+1 >= len(messages) {
			break
		}
		if messages[i].Role == "user" && messages[i+1].Role == "assistant" {
			totalPairs++
		}
	}

	currentPair := 0
	for i := 0; i < len(messages)-1; i += 2 {
		if i+1 >= len(messages) {
			break
		}

		userMsg := messages[i]
		assistantMsg := messages[i+1]

		if userMsg.Role != "user" || assistantMsg.Role != "assistant" {
			continue
		}

		currentPair++
		if progressChan != nil {
			progressChan <- vectorizeProgressMsg{current: currentPair, total: totalPairs}
		}

		content := fmt.Sprintf("Q: %s\nA: %s", userMsg.Content, assistantMsg.Content)
		mainChunkID := ""

		// Detect content type first (skip in light mode to save LLM calls)
		contentType := ContentType("dialog")
		if m.config.VectorExtractMetadata && !m.config.VectorLightMode {
			if progressChan != nil {
				progressChan <- vectorizeStepMsg{step: "Detecting content type"}
			}
			detectedType, _ := m.client.DetectContentType(m.config.Model, userMsg.Content, assistantMsg.Content)
			if detectedType != "" {
				contentType = ContentType(detectedType)
			}
		}

		// STRATEGY 1: Create main full Q&A chunk
		if progressChan != nil {
			progressChan <- vectorizeStepMsg{step: "Creating full Q&A chunk"}
		}
		embedding, err := m.client.GenerateEmbedding(m.config.VectorModel, content)
		if err != nil {
			return err
		}

		baseMetadata := ChunkMetadata{
			UserMessage:      userMsg.Content,
			AssistantMessage: assistantMsg.Content,
			Timestamp:        userMsg.Timestamp,
			ParentChunkID:    prevChunkID,
			OriginalText:     assistantMsg.Content,
		}

		mainChunk := VectorChunk{
			ChatID:      m.currentChat.ID,
			Content:     content,
			ContentType: contentType,
			Strategy:    StrategyFullQA,
			Embedding:   embedding,
			Metadata:    baseMetadata,
		}

		if err := m.vectorDB.AddChunk(mainChunk); err != nil {
			return err
		}

		mainChunkID = mainChunk.ID
		prevChunkID = mainChunk.ID
		relatedIDs := []string{}

		// ALWAYS apply sentence-level chunking (doesn't require LLM extraction)
		// STRATEGY 2: Sentence-level chunking for long responses
		if progressChan != nil {
			progressChan <- vectorizeStepMsg{step: "Chunking sentences"}
		}
		sentences := strings.Split(assistantMsg.Content, ". ")
		if len(sentences) > 1 {
			for idx, sentence := range sentences {
				sentence = strings.TrimSpace(sentence)
				if len(sentence) < 20 {
					continue
				}
				if !strings.HasSuffix(sentence, ".") && idx < len(sentences)-1 {
					sentence += "."
				}

				sentContent := fmt.Sprintf("Q: %s\nA: %s", userMsg.Content, sentence)
				if sentEmbed, err := m.client.GenerateEmbedding(m.config.VectorModel, sentContent); err == nil {
					sentChunk := VectorChunk{
						ChatID:      m.currentChat.ID,
						Content:     sentContent,
						ContentType: contentType,
						Strategy:    StrategySentence,
						Embedding:   sentEmbed,
						Metadata: ChunkMetadata{
							UserMessage:      userMsg.Content,
							AssistantMessage: sentence,
							Timestamp:        userMsg.Timestamp,
							ParentChunkID:    mainChunkID,
							SentenceIndex:    idx,
							OriginalText:     assistantMsg.Content,
						},
					}
					if err := m.vectorDB.AddChunk(sentChunk); err == nil {
						relatedIDs = append(relatedIDs, sentChunk.ID)
					}
				}
			}
		}

		// Apply advanced extraction strategies if enabled
		if m.config.VectorExtractMetadata {
			// STRATEGY 3: Extract structured who/what/why/when/where/how (skip in light mode)
			if !m.config.VectorLightMode {
				if progressChan != nil {
					progressChan <- vectorizeStepMsg{step: "Extracting structured Q&A"}
				}
				if structuredQA, err := m.client.ExtractStructuredQA(m.config.Model, userMsg.Content, assistantMsg.Content); err == nil && structuredQA != nil {
				qaContent := fmt.Sprintf("Who: %s\nWhat: %s\nWhy: %s\nWhen: %s\nWhere: %s\nHow: %s",
					structuredQA.Who, structuredQA.What, structuredQA.Why, structuredQA.When, structuredQA.Where, structuredQA.How)

				if qaEmbed, err := m.client.GenerateEmbedding(m.config.VectorModel, qaContent); err == nil {
					qaChunk := VectorChunk{
						ChatID:      m.currentChat.ID,
						Content:     qaContent,
						ContentType: contentType,
						Strategy:    StrategyWhoWhatWhy,
						Embedding:   qaEmbed,
						Metadata: ChunkMetadata{
							UserMessage:      userMsg.Content,
							AssistantMessage: assistantMsg.Content,
							Timestamp:        userMsg.Timestamp,
							ParentChunkID:    mainChunkID,
							Who:              structuredQA.Who,
							What:             structuredQA.What,
							Why:              structuredQA.Why,
							When:             structuredQA.When,
							Where:            structuredQA.Where,
							How:              structuredQA.How,
							SearchKeywords:   structuredQA.Keywords,
						},
					}
					if err := m.vectorDB.AddChunk(qaChunk); err == nil {
						relatedIDs = append(relatedIDs, qaChunk.ID)
					}
				}
				}
			}

			// STRATEGY 4: Extract key-value pairs (entity registry) - ALWAYS run, even in light mode
			if progressChan != nil {
				progressChan <- vectorizeStepMsg{step: "Extracting key-value pairs"}
			}
			if kvPairs, err := m.client.ExtractKeyValuePairs(m.config.Model, userMsg.Content, assistantMsg.Content); err == nil && len(kvPairs) > 0 {
				for _, kv := range kvPairs {
					kvContent := fmt.Sprintf("%s: %s", kv.Key, kv.Value)
					if kvEmbed, err := m.client.GenerateEmbedding(m.config.VectorModel, kvContent); err == nil {
						kvChunk := VectorChunk{
							ChatID:      m.currentChat.ID,
							Content:     kvContent,
							ContentType: contentType,
							Strategy:    StrategyKeyValue,
							Embedding:   kvEmbed,
							Metadata: ChunkMetadata{
								UserMessage:      userMsg.Content,
								AssistantMessage: kv.Value,
								Timestamp:        userMsg.Timestamp,
								ParentChunkID:    mainChunkID,
								EntityKey:        kv.Key,
								EntityValue:      kv.Value,
								SearchKeywords:   kv.Keywords,
							},
						}
						if err := m.vectorDB.AddChunk(kvChunk); err == nil {
							relatedIDs = append(relatedIDs, kvChunk.ID)
						}
					}
				}
			}

			// STRATEGY 5: Extract entity sheets for fictional content (skip in light mode)
			if contentType == ContentTypeFictional && !m.config.VectorLightMode {
				if progressChan != nil {
					progressChan <- vectorizeStepMsg{step: "Extracting entity sheets"}
				}
				if entities, err := m.client.ExtractEntitySheets(m.config.Model, userMsg.Content, assistantMsg.Content); err == nil && len(entities) > 0 {
					for _, entity := range entities {
						sheetContent := fmt.Sprintf("%s (%s): %s", entity.EntityName, entity.EntityType, entity.Description)
						for k, v := range entity.Attributes {
							sheetContent += fmt.Sprintf("\n%s: %s", k, v)
						}

						if sheetEmbed, err := m.client.GenerateEmbedding(m.config.VectorModel, sheetContent); err == nil {
							sheetChunk := VectorChunk{
								ChatID:      m.currentChat.ID,
								Content:     sheetContent,
								ContentType: contentType,
								Strategy:    StrategyEntitySheet,
								Embedding:   sheetEmbed,
								Metadata: ChunkMetadata{
									UserMessage:      userMsg.Content,
									AssistantMessage: sheetContent,
									Timestamp:        userMsg.Timestamp,
									ParentChunkID:    mainChunkID,
									EntityKey:        entity.EntityName,
									EntityValue:      sheetContent,
									SearchKeywords:   entity.Keywords,
									CharacterRefs:    []string{entity.EntityName},
								},
							}
							if err := m.vectorDB.AddChunk(sheetChunk); err == nil {
								relatedIDs = append(relatedIDs, sheetChunk.ID)
							}
						}
					}
				}
			}

			// STRATEGY 6: Extract canonical Q&A pairs - ALWAYS run, even in light mode
			if progressChan != nil {
				progressChan <- vectorizeStepMsg{step: "Extracting canonical Q&A"}
			}
			if canonicalQAs, err := m.client.ExtractCanonicalQA(m.config.Model, userMsg.Content, assistantMsg.Content); err == nil && len(canonicalQAs) > 0 {
				// Store canonical questions in the main chunk
				if mainChunk := m.vectorDB.GetChunkByID(mainChunkID); mainChunk != nil {
					questions := make([]string, len(canonicalQAs))
					for i, qa := range canonicalQAs {
						questions[i] = qa.Question
					}
					mainChunk.CanonicalQuestions = questions
					mainChunk.CanonicalAnswer = canonicalQAs[0].Answer // Use first answer as primary
					m.vectorDB.saveChunk(*mainChunk)
				}
			}

			// STRATEGY 7: Question-as-key chunks - ALWAYS run
			// Generate questions that would lead to this content
			if progressChan != nil {
				progressChan <- vectorizeStepMsg{step: "Generating question keys"}
			}
			if questionKeys, err := m.client.ExtractQuestionKeys(m.config.Model, userMsg.Content, assistantMsg.Content); err == nil && len(questionKeys) > 0 {
				// Create a separate chunk for each generated question
				// The question is the searchable content, full answer is referenced
				for _, qk := range questionKeys {
					qkContent := qk.Question
					if qkEmbed, err := m.client.GenerateEmbedding(m.config.VectorModel, qkContent); err == nil {
						qkChunk := VectorChunk{
							ChatID:      m.currentChat.ID,
							Content:     qkContent, // Question is the searchable content
							ContentType: contentType,
							Strategy:    StrategyQuestionKey,
							Embedding:   qkEmbed,
							Metadata: ChunkMetadata{
								UserMessage:      userMsg.Content,
								AssistantMessage: assistantMsg.Content, // Full content stored here
								Timestamp:        userMsg.Timestamp,
								ParentChunkID:    mainChunkID,
								SearchKeywords:   qk.Keywords,
								OriginalText:     assistantMsg.Content,
							},
						}
						// Store canonical question/answer pair
						qkChunk.CanonicalQuestions = []string{qk.Question}
						qkChunk.CanonicalAnswer = assistantMsg.Content

						if err := m.vectorDB.AddChunk(qkChunk); err == nil {
							relatedIDs = append(relatedIDs, qkChunk.ID)
						}
					}
				}
			}
		}

		// Link all sub-chunks to main chunk
		if mainChunk := m.vectorDB.GetChunkByID(mainChunkID); mainChunk != nil {
			mainChunk.Metadata.RelatedChunkIDs = relatedIDs
			m.vectorDB.saveChunk(*mainChunk)
		}
	}

	return nil
}

// retrieveRelevantContext searches vector DB for relevant past conversations
func (m *model) retrieveRelevantContext(query string) (string, error) {
	result, err := m.ragEngine.RetrieveContext(query)
	if err != nil {
		return "", err
	}

	// Update model state with results
	m.lastVectorResults = result.Results
	m.vectorContextUsed = result.ContextUsed
	m.lastVectorDebug = result.DebugInfo
	m.lastRAGResult = result // Store for refinement
	m.lastUserQuery = query  // Store for refinement

	return result.Context, nil
}

func (m *model) handleVectorStatsViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		m.currentView = chatListView
		m.vectorStatsScroll = 0
		return m, nil
	case "d":
		m.config.VectorDebug = !m.config.VectorDebug
		m.config.Save()
		return m, nil
	case "up", "k":
		if m.vectorStatsScroll > 0 {
			m.vectorStatsScroll--
		}
		return m, nil
	case "down", "j":
		m.vectorStatsScroll++
		return m, nil
	case "pgup":
		m.vectorStatsScroll -= 10
		if m.vectorStatsScroll < 0 {
			m.vectorStatsScroll = 0
		}
		return m, nil
	case "pgdown":
		m.vectorStatsScroll += 10
		return m, nil
	case "home":
		m.vectorStatsScroll = 0
		return m, nil
	}
	return m, nil
}

func (m model) renderVectorStatsView() string {
	title := titleStyle.Render("Vector Database Statistics & Debug")
	help := helpStyle.Render("↑/↓/PgUp/PgDn: scroll | d: toggle debug | esc: back")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	if !m.config.VectorEnabled {
		content.WriteString("Vector DB is currently disabled.\n")
		content.WriteString("Enable it in Settings to start building knowledge.\n\n")
		content.WriteString(help)
		return content.String()
	}

	stats := m.vectorDB.GetStats()

	content.WriteString(helpStyle.Render("Database Stats:") + "\n")
	content.WriteString(fmt.Sprintf("  Total Chunks: %v\n", stats["total_chunks"]))
	content.WriteString(fmt.Sprintf("  Unique Chats: %v\n", stats["unique_chats"]))
	content.WriteString(fmt.Sprintf("  Marked Bad: %v\n", stats["marked_bad"]))
	content.WriteString(fmt.Sprintf("  Verified: %v\n", stats["verified"]))
	content.WriteString(fmt.Sprintf("  Storage: %v\n\n", stats["storage_path"]))

	content.WriteString(helpStyle.Render("Configuration:") + "\n")
	content.WriteString(fmt.Sprintf("  Model: %s\n", m.config.VectorModel))
	content.WriteString(fmt.Sprintf("  Top-K Results: %d\n", m.config.VectorTopK))
	content.WriteString(fmt.Sprintf("  Similarity Threshold: %.2f\n", m.config.VectorSimilarity))

	debugStatus := "OFF"
	if m.config.VectorDebug {
		debugStatus = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("ON")
	}
	content.WriteString(fmt.Sprintf("  Debug Mode: %s\n", debugStatus))

	extractStatus := "OFF"
	if m.config.VectorExtractMetadata {
		extractStatus = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("ON")
	}
	content.WriteString(fmt.Sprintf("  Metadata Extraction: %s\n", extractStatus))

	relatedStatus := "OFF"
	if m.config.VectorIncludeRelated {
		relatedStatus = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("ON")
	}
	content.WriteString(fmt.Sprintf("  Include Related Chunks: %s\n\n", relatedStatus))

	// Extraction stats
	extractStats := m.client.GetExtractionStats()
	if len(extractStats) > 0 {
		content.WriteString(helpStyle.Render("Extraction Statistics:") + "\n")

		totalSuccess := extractStats["structured_qa_success"] + extractStats["kv_pairs_success"] +
			extractStats["entity_sheets_success"] + extractStats["canonical_qa_success"] + extractStats["question_keys_success"]
		totalFailed := extractStats["structured_qa_failed"] + extractStats["kv_pairs_failed"] +
			extractStats["entity_sheets_failed"] + extractStats["canonical_qa_failed"] + extractStats["question_keys_failed"]

		successStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
		failStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("196"))

		content.WriteString(fmt.Sprintf("  Total: %s / %s\n",
			successStyle.Render(fmt.Sprintf("%d success", totalSuccess)),
			failStyle.Render(fmt.Sprintf("%d failed", totalFailed))))

		if extractStats["structured_qa_success"] > 0 || extractStats["structured_qa_failed"] > 0 {
			content.WriteString(fmt.Sprintf("  Structured Q&A: %d / %d\n",
				extractStats["structured_qa_success"], extractStats["structured_qa_failed"]))
		}
		if extractStats["kv_pairs_success"] > 0 || extractStats["kv_pairs_failed"] > 0 {
			content.WriteString(fmt.Sprintf("  Key-Value Pairs: %d / %d\n",
				extractStats["kv_pairs_success"], extractStats["kv_pairs_failed"]))
		}
		if extractStats["entity_sheets_success"] > 0 || extractStats["entity_sheets_failed"] > 0 {
			content.WriteString(fmt.Sprintf("  Entity Sheets: %d / %d\n",
				extractStats["entity_sheets_success"], extractStats["entity_sheets_failed"]))
		}
		if extractStats["canonical_qa_success"] > 0 || extractStats["canonical_qa_failed"] > 0 {
			content.WriteString(fmt.Sprintf("  Canonical Q&A: %d / %d\n",
				extractStats["canonical_qa_success"], extractStats["canonical_qa_failed"]))
		}
		if extractStats["question_keys_success"] > 0 || extractStats["question_keys_failed"] > 0 {
			content.WriteString(fmt.Sprintf("  Question Keys: %d / %d\n",
				extractStats["question_keys_success"], extractStats["question_keys_failed"]))
		}

		lastError := m.client.GetLastError()
		if lastError != "" {
			content.WriteString("\n" + helpStyle.Render("Last Error:") + "\n")
			errorStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
			content.WriteString(errorStyle.Render(lastError) + "\n")
		}
		content.WriteString("\n")
	}

	if m.lastVectorDebug != "" {
		content.WriteString(helpStyle.Render("Last Query Debug:") + "\n")
		debugStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("243"))
		content.WriteString(debugStyle.Render(m.lastVectorDebug))
		content.WriteString("\n")
	} else {
		content.WriteString("No recent vector queries.\n")
		content.WriteString("Send a message to see debug info here.\n\n")
	}

	content.WriteString(help)

	// Apply scrolling
	fullContent := content.String()
	lines := strings.Split(fullContent, "\n")

	// Calculate visible window
	maxLines := m.height - 2 // Leave room for borders
	if maxLines < 10 {
		maxLines = 10
	}

	startLine := m.vectorStatsScroll
	if startLine >= len(lines) {
		startLine = len(lines) - 1
		if startLine < 0 {
			startLine = 0
		}
	}

	endLine := startLine + maxLines
	if endLine > len(lines) {
		endLine = len(lines)
	}

	// Show scroll indicator if content is larger than viewport
	visibleLines := lines[startLine:endLine]
	result := strings.Join(visibleLines, "\n")

	if len(lines) > maxLines {
		scrollInfo := fmt.Sprintf("\n[Line %d-%d of %d]", startLine+1, endLine, len(lines))
		result += helpStyle.Render(scrollInfo)
	}

	return result
}

func truncateString(s string, maxLen int) string {
	// Flatten newlines and normalize whitespace
	flattened := strings.ReplaceAll(s, "\n", " ")
	flattened = strings.ReplaceAll(flattened, "\r", " ")
	flattened = strings.Join(strings.Fields(flattened), " ")

	if len(flattened) <= maxLen {
		return flattened
	}
	return flattened[:maxLen] + "..."
}

func (m *model) handleConfirmResetViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "y", "Y":
		// Perform reset
		return m, m.resetAllData

	case "n", "N", "esc", "q":
		m.currentView = chatListView
		return m, nil
	}
	return m, nil
}

func (m model) renderConfirmResetView() string {
	title := errorStyle.Render("RESET ALL DATA")
	help := helpStyle.Render("y: confirm reset | n/esc: cancel")

	var content strings.Builder
	content.WriteString(title + "\n\n")

	warningStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Bold(true)
	content.WriteString(warningStyle.Render("WARNING: This will permanently delete:") + "\n\n")

	content.WriteString("  - All chat conversations\n")
	content.WriteString("  - All vector database embeddings\n")
	content.WriteString("  - All indexed knowledge\n\n")

	infoStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("86"))
	content.WriteString(infoStyle.Render("Preserved:") + "\n\n")
	content.WriteString("  - Configuration settings\n")
	content.WriteString("  - Backups (if any)\n\n")

	content.WriteString(errorStyle.Render("This action cannot be undone!") + "\n\n")
	content.WriteString(help)

	return content.String()
}

func (m *model) resetAllData() tea.Msg {
	// Clear all chats
	if err := m.storage.ClearAllChats(); err != nil {
		return errMsg{err: fmt.Errorf("failed to clear chats: %w", err)}
	}

	// Clear vector DB
	if err := m.vectorDB.ClearAll(); err != nil {
		return errMsg{err: fmt.Errorf("failed to clear vector DB: %w", err)}
	}

	return resetCompleteMsg{}
}

// Document Import View
func (m model) renderDocumentImportView() string {
	title := titleStyle.Render("Document Import - Build Knowledge Base")
	var help string
	if m.importing {
		help = helpStyle.Render("c or esc: cancel import")
	} else {
		help = helpStyle.Render("↑/↓: navigate | enter: import | a: import all | esc: back")
	}

	var content strings.Builder
	content.WriteString(title + "\n\n")
	content.WriteString(helpStyle.Render(fmt.Sprintf("Path: %s", m.importPath)) + "\n\n")

	if m.importing {
		if m.importCancelled {
			content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("214")).Render("Cancelling...") + "\n\n")
		} else {
			content.WriteString("Importing...\n\n")
		}
		for _, line := range m.importProgress {
			content.WriteString(fmt.Sprintf("  %s\n", line))
		}
		content.WriteString("\n" + help)
		return content.String()
	}

	// Show import results if available (after import completes)
	if len(m.importProgress) > 0 {
		content.WriteString("Import Complete\n\n")
		for _, line := range m.importProgress {
			content.WriteString(fmt.Sprintf("  %s\n", line))
		}
		content.WriteString("\n" + helpStyle.Render("esc: close"))
		return content.String()
	}

	if len(m.scannedFiles) == 0 {
		content.WriteString(helpStyle.Render("No supported files found. Scanning for .md, .go, .ts, .js, .py, .rs files...") + "\n")
	} else {
		content.WriteString(helpStyle.Render(fmt.Sprintf("Found %d files:", len(m.scannedFiles))) + "\n")

		// Calculate display window
		maxVisible := 20
		displayStart := 0
		displayEnd := len(m.scannedFiles)

		if len(m.scannedFiles) > maxVisible {
			// Center cursor in window
			displayStart = m.importCursor - maxVisible/2
			if displayStart < 0 {
				displayStart = 0
			}
			displayEnd = displayStart + maxVisible
			if displayEnd > len(m.scannedFiles) {
				displayEnd = len(m.scannedFiles)
				displayStart = displayEnd - maxVisible
				if displayStart < 0 {
					displayStart = 0
				}
			}
		}

		for i := displayStart; i < displayEnd; i++ {
			// Skip invalid entries
			if i >= len(m.scannedFiles) || m.scannedFiles[i] == "" {
				continue
			}

			cursor := " "
			if i == m.importCursor {
				cursor = ">"
			}

			// Show relative path or basename for cleaner display
			displayPath := m.scannedFiles[i]
			if relPath, err := filepath.Rel(m.importPath, displayPath); err == nil && relPath != "" && relPath != "." {
				displayPath = relPath
			} else {
				// Fallback to basename if Rel fails
				displayPath = filepath.Base(displayPath)
			}

			// Skip empty paths
			if displayPath == "" || displayPath == "." {
				displayPath = m.scannedFiles[i]
			}

			// Get file size
			sizeStr := ""
			if info, err := os.Stat(m.scannedFiles[i]); err == nil {
				size := info.Size()
				if size < 1024 {
					sizeStr = fmt.Sprintf("%dB", size)
				} else if size < 1024*1024 {
					sizeStr = fmt.Sprintf("%.1fK", float64(size)/1024)
				} else {
					sizeStr = fmt.Sprintf("%.1fM", float64(size)/(1024*1024))
				}
			}

			// Replace tabs and newlines in displayPath to prevent layout issues
			displayPath = strings.ReplaceAll(displayPath, "\t", " ")
			displayPath = strings.ReplaceAll(displayPath, "\n", " ")
			displayPath = strings.ReplaceAll(displayPath, "\r", " ")

			fileLine := fmt.Sprintf("%-2s%-50s %8s", cursor, displayPath, sizeStr)
			if i == m.importCursor {
				fileLine = "\033[38;5;205m" + fileLine + "\033[0m"
			}
			content.WriteString(fileLine + "\n")
		}

		content.WriteString(helpStyle.Render(fmt.Sprintf("\nShowing %d-%d of %d", displayStart+1, displayEnd, len(m.scannedFiles))) + "\n")
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleDocumentImportViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		if m.importing {
			// Cancel import
			m.importCancelled = true
			if m.importCancelChan != nil {
				m.importCancelChan <- true
			}
			return m, nil
		}
		// Clear import progress when closing results
		if len(m.importProgress) > 0 {
			m.importProgress = []string{}
		}
		m.currentView = chatListView
		return m, m.loadChats

	case "c", "C":
		if m.importing {
			// Cancel import
			m.importCancelled = true
			if m.importCancelChan != nil {
				m.importCancelChan <- true
			}
		}

	case "up", "k":
		if !m.importing && m.importCursor > 0 {
			m.importCursor--
		}

	case "down", "j":
		if !m.importing && m.importCursor < len(m.scannedFiles)-1 {
			m.importCursor++
		}

	case "enter":
		if len(m.scannedFiles) > 0 && !m.importing {
			// Show strategy selection
			m.importAll = false
			m.strategyCursor = 0
			m.currentView = strategySelectionView
			return m, nil
		}

	case "a", "A":
		if len(m.scannedFiles) > 0 && !m.importing {
			// Show strategy selection
			m.importAll = true
			m.strategyCursor = 0
			m.currentView = strategySelectionView
			return m, nil
		}
	}

	return m, nil
}

func (m *model) scanDirectory() tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		files, err := m.docImporter.ScanDirectory(m.importPath)
		if err != nil {
			return errMsg{err: err}
		}

		return scanCompleteMsg{files: files}
	}
}

type importProgressMsg struct {
	message string
}

type importCompleteMsg struct{}

type scanCompleteMsg struct {
	files []string
}

func (m *model) importDocument(filePath string) tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		// Start import in goroutine
		go func() {
			m.importProgressChan <- fmt.Sprintf("[1/1] %s", filepath.Base(filePath))
			err := m.docImporter.ImportDocument(filePath, m.config.Model, m.config.VectorModel, m.importProgressChan)

			if err != nil {
				m.importProgressChan <- fmt.Sprintf("Error: %v", err)
			} else {
				m.importProgressChan <- "Complete!"
			}
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}

func (m *model) waitForImportProgress(progressChan chan string) tea.Cmd {
	return func() tea.Msg {
		if progressChan == nil {
			return importCompleteMsg{}
		}
		msg, ok := <-progressChan
		if !ok {
			// Channel closed, import done
			return importCompleteMsg{}
		}
		// Send progress and continue listening
		return importProgressMsg{message: msg}
	}
}

func (m *model) importAllDocuments() tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		totalFiles := len(m.scannedFiles)

		// Start import in goroutine
		go func() {
			chunksBefore := len(m.vectorDB.GetAllChunks())
			imported := 0
			skipped := 0
			failed := 0

			for i, filePath := range m.scannedFiles {
				m.importProgressChan <- fmt.Sprintf("[%d/%d] %s", i+1, totalFiles, filepath.Base(filePath))

				err := m.docImporter.ImportDocument(filePath, m.config.Model, m.config.VectorModel, m.importProgressChan)
				if err != nil {
					if strings.Contains(err.Error(), "already imported") {
						skipped++
					} else {
						failed++
						m.importProgressChan <- fmt.Sprintf("  Error: %v", err)
					}
				} else {
					imported++
				}
			}

			chunksAfter := len(m.vectorDB.GetAllChunks())
			newChunks := chunksAfter - chunksBefore

			summary := fmt.Sprintf("\nComplete! Files: %d imported, %d skipped, %d failed | New chunks: %d",
				imported, skipped, failed, newChunks)
			m.importProgressChan <- summary

			// Give UI time to display final message before closing
			time.Sleep(100 * time.Millisecond)
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}

// Strategy Selection View
func (m model) renderStrategySelectionView() string {
	title := titleStyle.Render("Select Import Strategies")
	help := helpStyle.Render("↑/↓: navigate | space: toggle | enter: confirm | esc: back")

	strategies := []struct {
		name        string
		description string
	}{
		{"all", "All 16 strategies (comprehensive, slow)"},
		{"basic", "Basic content only (fast)"},
		{"entity_sheet", "Character/location sheets"},
		{"who_what_why", "Structured Q&A extraction"},
		{"keyword", "Keywords and phrases"},
		{"sentence", "Sentence-level chunks"},
		{"full_qa", "Full Q&A pairs"},
		{"relationship_mapping", "Character relationships"},
		{"timeline", "Timeline and chronology"},
		{"conflict_plot", "Conflicts and plot points"},
		{"rule_mechanic", "Rules and mechanics"},
		{"project_planning", "Project planning"},
		{"requirements", "Requirements and specs"},
		{"task_breakdown", "Task breakdown"},
		{"document_section", "Document sections (markdown)"},
		{"code_snippet", "Code snippets"},
		{"tags", "Tag extraction (#tag)"},
		{"cross_references", "Document cross-references"},
	}

	var content strings.Builder
	content.WriteString(title + "\n\n")

	if m.importAll {
		content.WriteString(helpStyle.Render(fmt.Sprintf("Importing all %d files", len(m.scannedFiles))) + "\n\n")
	} else {
		content.WriteString(helpStyle.Render(fmt.Sprintf("Importing: %s", filepath.Base(m.scannedFiles[m.importCursor]))) + "\n\n")
	}

	selectedCount := len(m.selectedStrategies)
	if selectedCount > 0 {
		content.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(fmt.Sprintf("Selected: %d strategies", selectedCount)) + "\n\n")
	} else {
		content.WriteString(helpStyle.Render("No strategies selected") + "\n\n")
	}

	for i, strategy := range strategies {
		cursor := "  "
		checkbox := "[ ]"

		if m.selectedStrategies[strategy.name] {
			checkbox = "[x]"
		}

		if i == m.strategyCursor {
			cursor = "> "
		}

		line := fmt.Sprintf("%s%s %-20s %s", cursor, checkbox, strategy.name, strategy.description)
		if i == m.strategyCursor {
			line = lipgloss.NewStyle().Foreground(lipgloss.Color("205")).Render(line)
		}
		content.WriteString(line + "\n")
	}

	content.WriteString("\n" + help)
	return content.String()
}

func (m *model) handleStrategySelectionViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	strategies := []string{
		"all", "basic", "entity_sheet", "who_what_why", "keyword",
		"sentence", "full_qa", "relationship_mapping", "timeline",
		"conflict_plot", "rule_mechanic", "project_planning",
		"requirements", "task_breakdown", "document_section",
		"code_snippet", "tags", "cross_references",
	}

	switch msg.String() {
	case "esc":
		m.currentView = documentImportView
		m.selectedStrategies = make(map[string]bool) // Clear selections
		return m, nil

	case "up", "k":
		if m.strategyCursor > 0 {
			m.strategyCursor--
		}

	case "down", "j":
		if m.strategyCursor < len(strategies)-1 {
			m.strategyCursor++
		}

	case " ": // Spacebar to toggle selection
		strategyName := strategies[m.strategyCursor]
		if m.selectedStrategies[strategyName] {
			delete(m.selectedStrategies, strategyName)
		} else {
			m.selectedStrategies[strategyName] = true
		}

	case "enter":
		// Start import with selected strategies
		if len(m.selectedStrategies) == 0 {
			return m, nil // Don't start if no strategies selected
		}

		m.currentView = documentImportView
		m.importing = true
		m.importCancelled = false
		m.importProgress = []string{}
		m.importProgressChan = make(chan string, 100)
		m.importCancelChan = make(chan bool, 1)

		if m.importAll {
			return m, m.importAllDocumentsWithStrategies(m.selectedStrategies)
		} else {
			return m, m.importDocumentWithStrategies(m.scannedFiles[m.importCursor], m.selectedStrategies)
		}
	}

	return m, nil
}

func (m *model) importDocumentWithStrategies(filePath string, strategies map[string]bool) tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		// Convert map to slice for display
		strategyList := make([]string, 0, len(strategies))
		for s := range strategies {
			strategyList = append(strategyList, s)
		}

		// Start import in goroutine
		go func() {
			m.importProgressChan <- fmt.Sprintf("[1/1] %s (strategies: %v)", filepath.Base(filePath), strategyList)

			for strategy := range strategies {
				m.importProgressChan <- fmt.Sprintf("  Applying strategy: %s", strategy)
				err := m.docImporter.ImportDocumentWithStrategy(filePath, m.config.Model, m.config.VectorModel, strategy, false, m.importProgressChan)

				if err != nil {
					m.importProgressChan <- fmt.Sprintf("  Strategy %s error: %v", strategy, err)
				}
			}

			m.importProgressChan <- "Complete!"
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}

func (m *model) importDocumentWithStrategy(filePath string, strategy string) tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		// Start import in goroutine
		go func() {
			m.importProgressChan <- fmt.Sprintf("[1/1] %s (strategy: %s)", filepath.Base(filePath), strategy)
			err := m.docImporter.ImportDocumentWithStrategy(filePath, m.config.Model, m.config.VectorModel, strategy, false, m.importProgressChan)

			if err != nil {
				m.importProgressChan <- fmt.Sprintf("Error: %v", err)
			} else {
				m.importProgressChan <- "Complete!"
			}
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}

func (m *model) importAllDocumentsWithStrategies(strategies map[string]bool) tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		totalFiles := len(m.scannedFiles)

		// Convert map to slice for display
		strategyList := make([]string, 0, len(strategies))
		for s := range strategies {
			strategyList = append(strategyList, s)
		}

		// Start import in goroutine
		go func() {
			chunksBefore := len(m.vectorDB.GetAllChunks())
			imported := 0
			skipped := 0
			failed := 0

			// Track stats per strategy
			strategySuccess := make(map[string]int)
			strategyFailed := make(map[string]int)
			strategySkipped := make(map[string]int)

			for i, filePath := range m.scannedFiles {
				// Check for cancellation
				select {
				case <-m.importCancelChan:
					m.importProgressChan <- "\nImport cancelled by user"
					time.Sleep(100 * time.Millisecond)
					close(m.importProgressChan)
					return
				default:
				}

				m.importProgressChan <- fmt.Sprintf("[%d/%d] %s (strategies: %v)", i+1, totalFiles, filepath.Base(filePath), strategyList)
				fileHadSuccess := false

				for strategy := range strategies {
					// Check for cancellation before each strategy
					select {
					case <-m.importCancelChan:
						m.importProgressChan <- "\nImport cancelled by user"
						time.Sleep(100 * time.Millisecond)
						close(m.importProgressChan)
						return
					default:
					}

					m.importProgressChan <- fmt.Sprintf("  Applying strategy: %s", strategy)
					err := m.docImporter.ImportDocumentWithStrategy(filePath, m.config.Model, m.config.VectorModel, strategy, false, m.importProgressChan)
					if err != nil {
						if strings.Contains(err.Error(), "already imported") {
							skipped++
							strategySkipped[strategy]++
						} else {
							failed++
							strategyFailed[strategy]++
							m.importProgressChan <- fmt.Sprintf("  Strategy %s error: %v", strategy, err)
						}
					} else {
						strategySuccess[strategy]++
						fileHadSuccess = true
					}
				}
				if fileHadSuccess {
					imported++
				}
			}

			chunksAfter := len(m.vectorDB.GetAllChunks())
			newChunks := chunksAfter - chunksBefore

			// Build detailed summary
			var summaryBuilder strings.Builder
			summaryBuilder.WriteString("\n========================================\n")
			summaryBuilder.WriteString("Import Summary\n")
			summaryBuilder.WriteString("========================================\n")
			summaryBuilder.WriteString(fmt.Sprintf("Files: %d imported, %d skipped, %d failed\n", imported, skipped, failed))
			summaryBuilder.WriteString(fmt.Sprintf("New chunks created: %d\n", newChunks))
			summaryBuilder.WriteString("\nStrategy Breakdown:\n")

			// Sort strategies for consistent display
			sortedStrategies := make([]string, 0, len(strategies))
			for s := range strategies {
				sortedStrategies = append(sortedStrategies, s)
			}
			sort.Strings(sortedStrategies)

			for _, strategy := range sortedStrategies {
				success := strategySuccess[strategy]
				skippedCount := strategySkipped[strategy]
				failedCount := strategyFailed[strategy]
				total := success + skippedCount + failedCount

				summaryBuilder.WriteString(fmt.Sprintf("  %-20s %d success, %d skipped, %d failed (of %d)\n",
					strategy+":", success, skippedCount, failedCount, total))
			}
			summaryBuilder.WriteString("========================================\n")
			summaryBuilder.WriteString("\nPress ESC to continue...")

			m.importProgressChan <- summaryBuilder.String()

			// Keep summary visible - don't close channel immediately
			// The channel will be closed when user presses ESC
			time.Sleep(5 * time.Second)
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}

func (m *model) importAllDocumentsWithStrategy(strategy string) tea.Cmd {
	return func() tea.Msg {
		if m.docImporter == nil {
			m.docImporter = NewDocumentImporter(m.client, m.vectorDB, m.importPath)
		}

		totalFiles := len(m.scannedFiles)

		// Start import in goroutine
		go func() {
			chunksBefore := len(m.vectorDB.GetAllChunks())
			imported := 0
			skipped := 0
			failed := 0

			for i, filePath := range m.scannedFiles {
				m.importProgressChan <- fmt.Sprintf("[%d/%d] %s (strategy: %s)", i+1, totalFiles, filepath.Base(filePath), strategy)

				err := m.docImporter.ImportDocumentWithStrategy(filePath, m.config.Model, m.config.VectorModel, strategy, false, m.importProgressChan)
				if err != nil {
					if strings.Contains(err.Error(), "already imported") {
						skipped++
					} else {
						failed++
						m.importProgressChan <- fmt.Sprintf("  Error: %v", err)
					}
				} else {
					imported++
				}
			}

			chunksAfter := len(m.vectorDB.GetAllChunks())
			newChunks := chunksAfter - chunksBefore

			summary := fmt.Sprintf("\nComplete! Files: %d imported, %d skipped, %d failed | New chunks: %d",
				imported, skipped, failed, newChunks)
			m.importProgressChan <- summary

			// Give UI time to display final message before closing
			time.Sleep(100 * time.Millisecond)
			close(m.importProgressChan)
		}()

		// Return first message to start the chain
		return m.waitForImportProgress(m.importProgressChan)()
	}
}
