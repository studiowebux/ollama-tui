package main

import (
	"fmt"
	"sort"
	"strings"

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
)

type model struct {
	storage           *Storage
	client            *OllamaClient
	config            *Config
	vectorDB          *VectorDB
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

	return model{
		storage:        storage,
		client:         client,
		config:         config,
		vectorDB:       vectorDB,
		projectManager: pm,
		currentView:    chatListView,
		textarea:      ta,
		viewport:      vp,
		messages:      []string{},
		endpointInput: endpointTa,
		summaryInput:  summaryTa,
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
		m.viewport.Height = msg.Height - 8
		m.textarea.SetWidth(msg.Width - 4)

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
		}

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
	}
	return ""
}

func (m model) renderChatView() string {
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

	help := helpStyle.Render("esc: back | ctrl+j/k or pgup/pgdn: scroll | ctrl+n: new | ctrl+s: settings | ctrl+t: summarize | ctrl+b: vectorize | ctrl+v: vector info")

	status := ""
	if m.vectorContextUsed {
		vectorIndicator := lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("Vector context used")
		status = vectorIndicator + " "
	}
	if m.vectorizing {
		if m.vectorProgress != "" {
			status += helpStyle.Render(fmt.Sprintf("Vectorizing conversation... %s", m.vectorProgress))
		} else {
			status += helpStyle.Render("Vectorizing conversation...")
		}
	} else if m.summarizing {
		status += helpStyle.Render("Summarizing conversation...")
	} else if m.streaming {
		status += helpStyle.Render("Streaming...")
	}
	if m.err != nil {
		status = errorStyle.Render(fmt.Sprintf("Error: %v", m.err))
	}

	var content strings.Builder
	content.WriteString(title + "\n\n")
	content.WriteString(m.viewport.View() + "\n\n")
	content.WriteString(m.textarea.View() + "\n")
	if status != "" {
		content.WriteString(status + "\n")
	}
	content.WriteString(help)

	return content.String()
}

func (m model) renderChatListView() string {
	projectName := "default"
	if project := m.projectManager.GetProject(m.config.CurrentProject); project != nil {
		projectName = project.Name
	}
	title := titleStyle.Render(fmt.Sprintf("Chat History - Project: %s", projectName))
	modelInfo := helpStyle.Render(fmt.Sprintf("Current model: %s", m.config.Model))
	help := helpStyle.Render("↑/↓: navigate | enter: open | n: new chat | d: delete | p: projects | k: knowledge base | s: settings | v: vector stats | r: reset all | q: quit")

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

	// Handle enter for sending messages
	if msg.Type == tea.KeyEnter {
		if m.streaming {
			return m, nil
		}
		return m, m.sendMessage()
	}

	// Pass all other keys to textarea when not streaming
	if !m.streaming {
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

	for i := 0; i < len(m.messages); i++ {
		role := "user"
		if i < len(m.messageRoles) {
			role = m.messageRoles[i]
		}

		if role == "user" {
			content.WriteString(userStyle.Render("You:") + "\n")
			content.WriteString(m.messages[i] + "\n\n")
		} else {
			content.WriteString(assistantStyle.Render("Assistant:") + "\n")
			content.WriteString(renderMessageWithThinking(m.messages[i]) + "\n\n")
		}
	}

	m.viewport.SetContent(content.String())
	m.viewport.GotoBottom()
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

	// Retrieve relevant context from vector DB
	relevantContext, err := m.retrieveRelevantContext(userMsg)
	if err != nil {
		return func() tea.Msg {
			return errMsg{err: fmt.Errorf("context retrieval failed: %w", err)}
		}
	}

	chatMessages := make([]ChatMessage, 0, len(m.currentChat.Messages))
	for _, msg := range m.currentChat.Messages {
		chatMessages = append(chatMessages, ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Prepend context to the last user message if found
	if relevantContext != "" {
		lastIdx := len(chatMessages) - 1
		chatMessages[lastIdx].Content = relevantContext + "\n---\n\n" + chatMessages[lastIdx].Content
	}

	return m.streamResponse(chatMessages)
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
	m.lastVectorResults = nil
	m.vectorContextUsed = false
	m.lastVectorDebug = ""

	if !m.config.VectorEnabled {
		m.lastVectorDebug = "[Vector DB disabled]"
		return "", nil
	}

	// Enhance query if explicitly enabled (disabled by default for speed)
	var searchQueries []string
	searchQueries = append(searchQueries, query)

	if m.config.VectorEnhanceQuery {
		if enhancement, err := m.client.EnhanceQuery(m.config.Model, query); err == nil && enhancement != nil {
			// Add canonical form
			if enhancement.CanonicalForm != "" {
				searchQueries = append(searchQueries, enhancement.CanonicalForm)
			}
			// Add enhanced queries (limit to 2 most relevant)
			for i, eq := range enhancement.EnhancedQueries {
				if i >= 2 {
					break
				}
				searchQueries = append(searchQueries, eq)
			}
		}
	}

	// Search with all query variations and combine results
	allResults := make(map[string]SearchResult)

	for _, sq := range searchQueries {
		embedding, err := m.client.GenerateEmbedding(m.config.VectorModel, sq)
		if err != nil {
			continue
		}

		// Use hybrid search for better keyword matching
		results := m.vectorDB.SearchHybrid(embedding, sq, m.config.VectorTopK*2)

		// Merge results, keeping highest similarity score for each chunk
		for _, result := range results {
			if existing, ok := allResults[result.Chunk.ID]; !ok || result.Similarity > existing.Similarity {
				allResults[result.Chunk.ID] = result
			}
		}
	}

	// Convert map to slice and sort by similarity
	results := make([]SearchResult, 0, len(allResults))
	for _, result := range allResults {
		results = append(results, result)
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Limit to initial topK
	if len(results) > m.config.VectorTopK*2 {
		results = results[:m.config.VectorTopK*2]
	}

	// Optionally expand with related chunks
	if m.config.VectorIncludeRelated && len(results) > 0 {
		expanded := make(map[string]SearchResult)
		for _, result := range results {
			expanded[result.Chunk.ID] = result

			// Add parent context
			if result.Chunk.Metadata.ParentChunkID != "" {
				parent := m.vectorDB.GetChunkByID(result.Chunk.Metadata.ParentChunkID)
				if parent != nil {
					expanded[parent.ID] = SearchResult{
						Chunk:      *parent,
						Similarity: result.Similarity * 0.9,
					}
				}
			}

			// Add related chunks
			for _, relatedID := range result.Chunk.Metadata.RelatedChunkIDs {
				related := m.vectorDB.GetChunkByID(relatedID)
				if related != nil {
					expanded[related.ID] = SearchResult{
						Chunk:      *related,
						Similarity: result.Similarity * 0.85,
					}
				}
			}
		}

		// Convert back to slice and sort
		results = make([]SearchResult, 0, len(expanded))
		for _, result := range expanded {
			results = append(results, result)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Similarity > results[j].Similarity
		})

		// Limit to topK after expansion
		if len(results) > m.config.VectorTopK {
			results = results[:m.config.VectorTopK]
		}
	}

	m.lastVectorResults = results

	var debugBuilder strings.Builder
	debugBuilder.WriteString(fmt.Sprintf("Query: %s\n", truncateString(query, 60)))
	debugBuilder.WriteString(fmt.Sprintf("Found %d results from vector DB\n", len(results)))

	if len(results) == 0 {
		m.lastVectorDebug = debugBuilder.String() + "No results found."
		return "", nil
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("Relevant context from past conversations:\n\n")
	usedCount := 0

	for i, result := range results {
		debugBuilder.WriteString(fmt.Sprintf("  %d. Similarity=%.4f (threshold=%.2f) ",
			i+1, result.Similarity, m.config.VectorSimilarity))

		if result.Similarity < m.config.VectorSimilarity {
			debugBuilder.WriteString("SKIPPED\n")
			debugBuilder.WriteString(fmt.Sprintf("     Q: %s\n", truncateString(result.Chunk.Metadata.UserMessage, 60)))
			debugBuilder.WriteString(fmt.Sprintf("     A: %s\n", truncateString(result.Chunk.Metadata.AssistantMessage, 60)))
			continue
		}

		debugBuilder.WriteString("USED\n")
		debugBuilder.WriteString(fmt.Sprintf("     Q: %s\n", truncateString(result.Chunk.Metadata.UserMessage, 60)))
		debugBuilder.WriteString(fmt.Sprintf("     A: %s\n", truncateString(result.Chunk.Metadata.AssistantMessage, 60)))
		usedCount++
		contextBuilder.WriteString(fmt.Sprintf("Q: %s\nA: %s\n\n",
			result.Chunk.Metadata.UserMessage,
			result.Chunk.Metadata.AssistantMessage))
	}

	debugBuilder.WriteString(fmt.Sprintf("\nTotal contexts injected: %d\n", usedCount))
	m.lastVectorDebug = debugBuilder.String()

	if usedCount > 0 {
		m.vectorContextUsed = true
		return contextBuilder.String(), nil
	}

	return "", nil
}

func (m *model) handleVectorStatsViewKeys(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "esc", "q":
		m.currentView = chatListView
		return m, nil
	case "d":
		m.config.VectorDebug = !m.config.VectorDebug
		m.config.Save()
		return m, nil
	}
	return m, nil
}

func (m model) renderVectorStatsView() string {
	title := titleStyle.Render("Vector Database Statistics & Debug")
	help := helpStyle.Render("esc: back | d: toggle debug mode")

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
	return content.String()
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
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
