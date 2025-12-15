package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// processRelationshipMapping extracts relationships between entities
func (di *DocumentImporter) processRelationshipMapping(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting entity relationships"
	}

	prompt := fmt.Sprintf(`Extract all relationships between entities (characters, locations, organizations, concepts) from this text.

For each relationship, identify:
1. Entity A (source)
2. Relationship type (is, has, controls, allies with, opposes, created by, located in, etc.)
3. Entity B (target)
4. Context/reason for the relationship

Text:
%s

Return ONLY a JSON array:
[{
  "entity_a": "Entity Name A",
  "relationship": "relationship type",
  "entity_b": "Entity Name B",
  "context": "explanation of relationship",
  "strength": "strong|medium|weak"
}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no relationships found")
	}

	var relationships []struct {
		EntityA      string `json:"entity_a"`
		Relationship string `json:"relationship"`
		EntityB      string `json:"entity_b"`
		Context      string `json:"context"`
		Strength     string `json:"strength"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &relationships); err != nil {
		return err
	}

	for _, rel := range relationships {
		// Create searchable content
		searchContent := fmt.Sprintf("%s %s %s %s", rel.EntityA, rel.Relationship, rel.EntityB, rel.Context)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     fmt.Sprintf("%s %s %s. %s", rel.EntityA, rel.Relationship, rel.EntityB, rel.Context),
			ContentType: ContentTypeFictional,
			Strategy:    "relationship_mapping",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				Entities:       []string{rel.EntityA, rel.EntityB},
				SearchKeywords: []string{rel.EntityA, rel.EntityB, rel.Relationship, rel.Strength},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What is the relationship between %s and %s?", rel.EntityA, rel.EntityB),
			fmt.Sprintf("How does %s relate to %s?", rel.EntityA, rel.EntityB),
			fmt.Sprintf("What is %s to %s?", rel.EntityA, rel.EntityB),
		}
		chunk.CanonicalAnswer = fmt.Sprintf("%s %s %s. %s", rel.EntityA, rel.Relationship, rel.EntityB, rel.Context)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processTimeline extracts chronological events
func (di *DocumentImporter) processTimeline(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting timeline and chronology"
	}

	prompt := fmt.Sprintf(`Extract all events from this text in chronological order.

For each event, identify:
1. When it happened (specific time/date or relative time like "before X", "during Y")
2. What happened (the event itself)
3. Who was involved
4. Where it happened
5. Significance/consequences

Text:
%s

Return ONLY a JSON array ordered chronologically:
[{
  "when": "time reference",
  "what": "event description",
  "who": "participants",
  "where": "location",
  "significance": "why this matters",
  "order": 1
}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no timeline events found")
	}

	var events []struct {
		When         string `json:"when"`
		What         string `json:"what"`
		Who          string `json:"who"`
		Where        string `json:"where"`
		Significance string `json:"significance"`
		Order        int    `json:"order"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &events); err != nil {
		return err
	}

	for _, event := range events {
		searchContent := fmt.Sprintf("%s %s %s %s %s", event.When, event.What, event.Who, event.Where, event.Significance)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     event.What,
			ContentType: ContentTypeFictional,
			Strategy:    "timeline",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				When:           event.When,
				What:           event.What,
				Who:            event.Who,
				Where:          event.Where,
				SearchKeywords: []string{event.When, event.Who, event.Where},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What happened %s?", event.When),
			fmt.Sprintf("When did %s happen?", event.What),
			fmt.Sprintf("What events involved %s?", event.Who),
		}
		chunk.CanonicalAnswer = fmt.Sprintf("%s: %s involving %s at %s. %s", event.When, event.What, event.Who, event.Where, event.Significance)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processConflictPlot extracts narrative conflicts and plot points
func (di *DocumentImporter) processConflictPlot(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting conflicts and plot points"
	}

	prompt := fmt.Sprintf(`Extract all conflicts, challenges, and plot points from this narrative.

For each conflict/plot point, identify:
1. Problem/conflict (what's the issue)
2. Stakes (what's at risk, why it matters)
3. Parties involved (who's in conflict)
4. Resolution status (resolved/ongoing/escalating)
5. Outcome/consequences (if resolved)

Text:
%s

Return ONLY a JSON array:
[{
  "problem": "the conflict or challenge",
  "stakes": "what's at risk",
  "parties": ["entity1", "entity2"],
  "status": "resolved|ongoing|escalating",
  "outcome": "what happened (if resolved)"
}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no conflicts found")
	}

	var conflicts []struct {
		Problem string   `json:"problem"`
		Stakes  string   `json:"stakes"`
		Parties []string `json:"parties"`
		Status  string   `json:"status"`
		Outcome string   `json:"outcome"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &conflicts); err != nil {
		return err
	}

	for _, conflict := range conflicts {
		searchContent := fmt.Sprintf("%s %s %s %s %s", conflict.Problem, conflict.Stakes, strings.Join(conflict.Parties, " "), conflict.Status, conflict.Outcome)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     conflict.Problem,
			ContentType: ContentTypeFictional,
			Strategy:    "conflict_plot",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				Entities:       conflict.Parties,
				SearchKeywords: append(conflict.Parties, conflict.Status, "conflict", "plot"),
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		questions := []string{
			fmt.Sprintf("What is the conflict involving %s?", strings.Join(conflict.Parties, " and ")),
			"What conflicts exist?",
		}
		if len(conflict.Parties) > 0 {
			questions = append(questions, fmt.Sprintf("What are the stakes for %s?", conflict.Parties[0]))
		}
		chunk.CanonicalQuestions = questions
		chunk.CanonicalAnswer = fmt.Sprintf("Problem: %s. Stakes: %s. Parties: %s. Status: %s. %s",
			conflict.Problem, conflict.Stakes, strings.Join(conflict.Parties, ", "), conflict.Status, conflict.Outcome)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processRuleMechanic extracts game rules, magic systems, world mechanics
func (di *DocumentImporter) processRuleMechanic(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting rules and mechanics"
	}

	prompt := fmt.Sprintf(`Extract all rules, mechanics, and systems from this text (game rules, magic systems, world laws, etc.).

For each rule/mechanic:
1. Rule name/title
2. Trigger/condition (when does it apply)
3. Effect/consequence (what happens)
4. Exceptions/limitations
5. Category (magic, physics, social, combat, etc.)

Text:
%s

Return ONLY a JSON array:
[{
  "name": "rule name",
  "trigger": "when this applies",
  "effect": "what happens",
  "exceptions": "limitations or exceptions",
  "category": "magic|physics|social|combat|economic|other"
}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no rules found")
	}

	var rules []struct {
		Name       string `json:"name"`
		Trigger    string `json:"trigger"`
		Effect     string `json:"effect"`
		Exceptions string `json:"exceptions"`
		Category   string `json:"category"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &rules); err != nil {
		return err
	}

	for _, rule := range rules {
		searchContent := fmt.Sprintf("%s %s %s %s %s", rule.Name, rule.Trigger, rule.Effect, rule.Exceptions, rule.Category)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     fmt.Sprintf("When %s, then %s", rule.Trigger, rule.Effect),
			ContentType: ContentTypeFictional,
			Strategy:    "rule_mechanic",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				EntityKey:      rule.Name,
				EntityValue:    fmt.Sprintf("Trigger: %s. Effect: %s. Exceptions: %s", rule.Trigger, rule.Effect, rule.Exceptions),
				RuleSystem:     rule.Category,
				SearchKeywords: []string{rule.Name, rule.Category, "rule", "mechanic", "system"},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What is %s?", rule.Name),
			fmt.Sprintf("How does %s work?", rule.Name),
			fmt.Sprintf("What happens when %s?", rule.Trigger),
		}
		chunk.CanonicalAnswer = fmt.Sprintf("%s: When %s, then %s. Exceptions: %s", rule.Name, rule.Trigger, rule.Effect, rule.Exceptions)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processProjectPlanning extracts project scope, requirements, and planning data
func (di *DocumentImporter) processProjectPlanning(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting project planning data"
	}

	prompt := fmt.Sprintf(`Extract project planning information from this document.

Identify:
1. Project goals/objectives
2. Scope (what's included)
3. Out of scope (what's excluded)
4. Key stakeholders/roles
5. Constraints (time, budget, technical)
6. Success criteria
7. Risks/dependencies

Text:
%s

Return ONLY JSON:
{
  "goals": ["goal1", "goal2"],
  "scope": ["item1", "item2"],
  "out_of_scope": ["item1"],
  "stakeholders": [{"role": "role", "name": "name"}],
  "constraints": {"time": "...", "budget": "...", "technical": "..."},
  "success_criteria": ["criterion1"],
  "risks": ["risk1"]
}`, doc.Content[:min(3000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, false)
	if jsonStr == "" {
		return fmt.Errorf("no project data found")
	}

	var project struct {
		Goals           []string          `json:"goals"`
		Scope           []string          `json:"scope"`
		OutOfScope      []string          `json:"out_of_scope"`
		Stakeholders    []struct {
			Role string `json:"role"`
			Name string `json:"name"`
		}                 `json:"stakeholders"`
		Constraints     map[string]string `json:"constraints"`
		SuccessCriteria []string          `json:"success_criteria"`
		Risks           []string          `json:"risks"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &project); err != nil {
		return err
	}

	// Create multiple chunks for different aspects

	// Goals chunk
	if len(project.Goals) > 0 {
		goalsContent := "Project Goals: " + strings.Join(project.Goals, "; ")
		embedding, _ := di.client.GenerateEmbedding(embedModel, goalsContent)

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     goalsContent,
			ContentType: ContentTypeFact,
			Strategy:    "project_planning",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				SearchKeywords: append([]string{"goals", "objectives", "project"}, project.Goals...),
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{"What are the project goals?", "What are we trying to achieve?"}
		chunk.CanonicalAnswer = goalsContent
		di.vectorDB.AddChunk(chunk)
	}

	// Scope chunk
	if len(project.Scope) > 0 {
		scopeContent := "In Scope: " + strings.Join(project.Scope, "; ")
		if len(project.OutOfScope) > 0 {
			scopeContent += ". Out of Scope: " + strings.Join(project.OutOfScope, "; ")
		}
		embedding, _ := di.client.GenerateEmbedding(embedModel, scopeContent)

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     scopeContent,
			ContentType: ContentTypeFact,
			Strategy:    "project_planning",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				SearchKeywords: []string{"scope", "in-scope", "out-of-scope", "project"},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{"What's in scope?", "What's out of scope?", "What are we building?"}
		chunk.CanonicalAnswer = scopeContent
		di.vectorDB.AddChunk(chunk)
	}

	// Risks chunk
	if len(project.Risks) > 0 {
		risksContent := "Project Risks: " + strings.Join(project.Risks, "; ")
		embedding, _ := di.client.GenerateEmbedding(embedModel, risksContent)

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     risksContent,
			ContentType: ContentTypeFact,
			Strategy:    "project_planning",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				SearchKeywords: []string{"risks", "dependencies", "blockers", "project"},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{"What are the risks?", "What could go wrong?", "What are the dependencies?"}
		chunk.CanonicalAnswer = risksContent
		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processRequirements extracts functional and non-functional requirements
func (di *DocumentImporter) processRequirements(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting requirements and specifications"
	}

	prompt := fmt.Sprintf(`Extract all requirements from this document.

Categorize each requirement as:
- Functional (what the system must do)
- Non-functional (performance, security, usability, etc.)
- Business (business rules, policies)
- Technical (technical constraints, integrations)

For each requirement:
1. Requirement ID or name
2. Category
3. Description
4. Priority (must-have, should-have, nice-to-have)
5. Acceptance criteria (how to verify)

Text:
%s

Return ONLY a JSON array:
[{
  "id": "REQ-001",
  "category": "functional|non-functional|business|technical",
  "description": "requirement description",
  "priority": "must-have|should-have|nice-to-have",
  "acceptance_criteria": "how to verify this requirement"
}]`, doc.Content[:min(3000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no requirements found")
	}

	var requirements []struct {
		ID                 string `json:"id"`
		Category           string `json:"category"`
		Description        string `json:"description"`
		Priority           string `json:"priority"`
		AcceptanceCriteria string `json:"acceptance_criteria"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &requirements); err != nil {
		return err
	}

	for _, req := range requirements {
		searchContent := fmt.Sprintf("%s %s %s %s %s", req.ID, req.Category, req.Description, req.Priority, req.AcceptanceCriteria)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     req.Description,
			ContentType: ContentTypeFact,
			Strategy:    "requirements",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				EntityKey:      req.ID,
				EntityValue:    req.Description,
				SearchKeywords: []string{req.ID, req.Category, req.Priority, "requirement"},
				Tags:           []string{req.Category, req.Priority},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What is %s?", req.ID),
			fmt.Sprintf("What are the %s requirements?", req.Category),
			fmt.Sprintf("What are the %s requirements?", req.Priority),
		}
		chunk.CanonicalAnswer = fmt.Sprintf("%s (%s, %s): %s. Acceptance: %s", req.ID, req.Category, req.Priority, req.Description, req.AcceptanceCriteria)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processTaskBreakdown extracts actionable tasks and work breakdown
func (di *DocumentImporter) processTaskBreakdown(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting tasks and work breakdown"
	}

	prompt := fmt.Sprintf(`Extract all actionable tasks and work items from this document.

For each task:
1. Task name/title
2. Description (what needs to be done)
3. Dependencies (what must be done first)
4. Estimated effort (if mentioned)
5. Assigned to (if mentioned)
6. Category (frontend, backend, design, testing, etc.)

Text:
%s

Return ONLY a JSON array:
[{
  "task": "task title",
  "description": "what needs to be done",
  "dependencies": ["task1", "task2"],
  "effort": "estimate if mentioned, otherwise empty",
  "assigned": "person/team if mentioned",
  "category": "frontend|backend|design|testing|devops|other"
}]`, doc.Content[:min(3000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no tasks found")
	}

	var tasks []struct {
		Task         string   `json:"task"`
		Description  string   `json:"description"`
		Dependencies []string `json:"dependencies"`
		Effort       string   `json:"effort"`
		Assigned     string   `json:"assigned"`
		Category     string   `json:"category"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &tasks); err != nil {
		return err
	}

	for _, task := range tasks {
		searchContent := fmt.Sprintf("%s %s %s %s %s", task.Task, task.Description, task.Category, task.Effort, task.Assigned)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     task.Description,
			ContentType: ContentTypeFact,
			Strategy:    "task_breakdown",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				EntityKey:      task.Task,
				EntityValue:    task.Description,
				SearchKeywords: append([]string{task.Category, task.Assigned, "task", "work"}, task.Dependencies...),
				Tags:           []string{task.Category},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What is the task '%s'?", task.Task),
			fmt.Sprintf("What %s tasks exist?", task.Category),
			fmt.Sprintf("What tasks are assigned to %s?", task.Assigned),
		}

		answer := fmt.Sprintf("Task: %s. %s", task.Task, task.Description)
		if len(task.Dependencies) > 0 {
			answer += fmt.Sprintf(" Dependencies: %s.", strings.Join(task.Dependencies, ", "))
		}
		if task.Effort != "" {
			answer += fmt.Sprintf(" Effort: %s.", task.Effort)
		}
		chunk.CanonicalAnswer = answer

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}
