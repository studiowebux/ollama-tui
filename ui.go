package main

import (
	"fmt"
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
)

type model struct {
	storage         *Storage
	client          *OllamaClient
	config          *Config
	currentView     view
	currentChat     *Chat
	chats           []*Chat
	textarea        textarea.Model
	viewport        viewport.Model
	messages        []string
	messageRoles    []string
	streaming       bool
	summarizing     bool
	err             error
	width           int
	height          int
	chatListCursor  int
	settingsInput   string
	settingsFocus   int
	models          []string
	modelCursor     int
	chunkChan       chan string
	errChan         chan error
	endpointInput   textarea.Model
	editingEndpoint bool
	summaryInput    textarea.Model
	editingSummary  bool
	contextSize     int
	lastKeyG        bool
}

type streamChunkMsg string
type streamDoneMsg struct{}
type streamStartMsg struct {
	chunkChan chan string
	errChan   chan error
}
type errMsg struct{ err error }
type contextSizeMsg int

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

func initialModel(storage *Storage, client *OllamaClient, config *Config) model {
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
		storage:       storage,
		client:        client,
		config:        config,
		currentView:   chatListView,
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

	help := helpStyle.Render("esc: back | ctrl+j/k or pgup/pgdn: scroll | ctrl+n: new | ctrl+s: settings | ctrl+t: summarize")

	status := ""
	if m.summarizing {
		status = helpStyle.Render("Summarizing conversation...")
	} else if m.streaming {
		status = helpStyle.Render("Streaming...")
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
	title := titleStyle.Render("Chat History")
	modelInfo := helpStyle.Render(fmt.Sprintf("Current model: %s", m.config.Model))
	help := helpStyle.Render("↑/↓: navigate | enter: open | n: new chat | d: delete | s: settings | q: quit")

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

	case "up", "k":
		if m.chatListCursor > 0 {
			m.chatListCursor--
		}

	case "down", "j":
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
		m.settingsFocus = (m.settingsFocus + 1) % 3

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

	chatMessages := make([]ChatMessage, 0, len(m.currentChat.Messages))
	for _, msg := range m.currentChat.Messages {
		chatMessages = append(chatMessages, ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
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
