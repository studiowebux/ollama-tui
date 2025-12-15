package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// extractStringValue converts interface{} to string, handling nested objects
func extractStringValue(val interface{}) string {
	if val == nil {
		return ""
	}
	switch v := val.(type) {
	case string:
		return v
	case map[string]interface{}:
		// If it's a nested object, convert to JSON string
		bytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(bytes)
	case []interface{}:
		// If it's an array, join elements
		parts := make([]string, 0, len(v))
		for _, item := range v {
			parts = append(parts, extractStringValue(item))
		}
		return strings.Join(parts, ", ")
	default:
		return fmt.Sprintf("%v", v)
	}
}

// fixCommonJSONIssues attempts to fix common JSON formatting errors from LLMs
func fixCommonJSONIssues(jsonStr string) string {
	// Remove trailing commas before closing brackets/braces
	jsonStr = strings.ReplaceAll(jsonStr, ",]", "]")
	jsonStr = strings.ReplaceAll(jsonStr, ",}", "}")
	jsonStr = strings.ReplaceAll(jsonStr, ", ]", "]")
	jsonStr = strings.ReplaceAll(jsonStr, ", }", "}")

	// Remove commas after colons with no value
	jsonStr = strings.ReplaceAll(jsonStr, ":,", ": null,")

	return jsonStr
}

// ProcessWithStrategy processes a document using the specified strategy
func (di *DocumentImporter) ProcessWithStrategy(doc ImportedDocument, strategy string, chatModel, embedModel string, progressChan chan<- string) error {
	switch strategy {
	case "all":
		return di.processAll(doc, chatModel, embedModel, progressChan)

	// Content strategies
	case "entity_sheet":
		return di.processEntitySheet(doc, chatModel, embedModel, progressChan)
	case "who_what_why":
		return di.processWhoWhatWhy(doc, chatModel, embedModel, progressChan)
	case "keyword":
		return di.processKeyword(doc, chatModel, embedModel, progressChan)
	case "sentence":
		return di.processSentence(doc, chatModel, embedModel, progressChan)
	case "full_qa":
		return di.processFullQA(doc, chatModel, embedModel, progressChan)
	case "document_section":
		return di.processMarkdown(doc, chatModel, embedModel, progressChan)
	case "code_snippet":
		return di.processCode(doc, chatModel, embedModel, progressChan)

	// Advanced narrative strategies
	case "relationship_mapping":
		return di.processRelationshipMapping(doc, chatModel, embedModel, progressChan)
	case "timeline":
		return di.processTimeline(doc, chatModel, embedModel, progressChan)
	case "conflict_plot":
		return di.processConflictPlot(doc, chatModel, embedModel, progressChan)
	case "rule_mechanic":
		return di.processRuleMechanic(doc, chatModel, embedModel, progressChan)

	// Project planning strategies
	case "project_planning":
		return di.processProjectPlanning(doc, chatModel, embedModel, progressChan)
	case "requirements":
		return di.processRequirements(doc, chatModel, embedModel, progressChan)
	case "task_breakdown":
		return di.processTaskBreakdown(doc, chatModel, embedModel, progressChan)

	// Relationship strategies
	case "tags":
		return di.processTags(doc, chatModel, embedModel, progressChan)
	case "cross_references":
		return di.processCrossReferences(doc, chatModel, embedModel, progressChan)

	default:
		return fmt.Errorf("unknown strategy: %s", strategy)
	}
}

// processAll applies multiple strategies for better retrieval
func (di *DocumentImporter) processAll(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Applying ALL 16 strategies for comprehensive coverage"
	}

	// Apply ALL strategies - no auto-detection
	strategies := []string{
		// Basic content strategies
		"entity_sheet",
		"who_what_why",
		"keyword",
		"sentence",
		"full_qa",
		// Advanced narrative strategies
		"relationship_mapping",
		"timeline",
		"conflict_plot",
		"rule_mechanic",
		// Project planning strategies
		"project_planning",
		"requirements",
		"task_breakdown",
		// Document structure strategies
		"document_section",
		"code_snippet",
		// Relationship strategies
		"tags",
		"cross_references",
	}

	if progressChan != nil {
		progressChan <- fmt.Sprintf("Will apply %d strategies", len(strategies))
	}

	for _, strategy := range strategies {
		if progressChan != nil {
			progressChan <- fmt.Sprintf("Strategy: %s", strategy)
		}
		if err := di.ProcessWithStrategy(doc, strategy, chatModel, embedModel, progressChan); err != nil {
			if progressChan != nil {
				progressChan <- fmt.Sprintf("Strategy %s failed: %v", strategy, err)
			}
			// Continue with other strategies
		}
	}

	return nil
}

// processEntitySheet creates character/location entity sheets
func (di *DocumentImporter) processEntitySheet(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting entities (characters, locations, items)"
	}

	prompt := fmt.Sprintf(`Extract all entities (characters, locations, items, factions) from this text.
For each entity, provide:
1. Entity name
2. Entity type (character/location/item/faction)
3. Full description/attributes

Text:
%s

Return ONLY a JSON array:
[{"name": "Entity Name", "type": "character", "description": "full description"}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no entities found")
	}

	var entities []struct {
		Name        string `json:"name"`
		Type        string `json:"type"`
		Description string `json:"description"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &entities); err != nil {
		return err
	}

	for _, entity := range entities {
		embedding, err := di.client.GenerateEmbedding(embedModel, entity.Description)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     entity.Description,
			ContentType: ContentTypeFictional,
			Strategy:    StrategyEntitySheet,
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				EntityKey:      entity.Name,
				EntityValue:    entity.Description,
				SearchKeywords: []string{entity.Name, entity.Type, "entity"},
				Entities:       []string{entity.Name},
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("Who is %s?", entity.Name),
			fmt.Sprintf("What is %s?", entity.Name),
			fmt.Sprintf("Tell me about %s", entity.Name),
		}
		chunk.CanonicalAnswer = entity.Description

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processWhoWhatWhy creates structured Q&A chunks
func (di *DocumentImporter) processWhoWhatWhy(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting structured Q&A (who/what/why/when/where/how)"
	}

	prompt := fmt.Sprintf(`Analyze this text and extract key information in structured format.
Provide: who (people/entities involved), what (what happened/is described), why (reasons/purpose),
when (time context), where (location), how (mechanism/method).

Text:
%s

Return ONLY a single JSON object (not an array). Format:
{"who": "description", "what": "description", "why": "description", "when": "description", "where": "description", "how": "description"}

If a field is not applicable, use an empty string "". Do not return an array.`, doc.Content[:min(2000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, false)
	if jsonStr == "" {
		// Try extracting as array if object extraction failed
		jsonStr = extractJSON(response, true)
		if jsonStr == "" {
			return fmt.Errorf("no structured data found in LLM response")
		}
	}

	var structured struct {
		Who   string `json:"who"`
		What  string `json:"what"`
		Why   string `json:"why"`
		When  string `json:"when"`
		Where string `json:"where"`
		How   string `json:"how"`
	}

	// Try parsing as object first
	if err := json.Unmarshal([]byte(jsonStr), &structured); err != nil {
		// If that fails, try parsing as array and take first element
		var arr []struct {
			Who   string `json:"who"`
			What  string `json:"what"`
			Why   string `json:"why"`
			When  string `json:"when"`
			Where string `json:"where"`
			How   string `json:"how"`
		}
		if err2 := json.Unmarshal([]byte(jsonStr), &arr); err2 != nil {
			// If both fail, try parsing with flexible types (handle nested objects)
			var flexible map[string]interface{}
			if err3 := json.Unmarshal([]byte(jsonStr), &flexible); err3 != nil {
				return fmt.Errorf("all parsing attempts failed - object: %v, array: %v, flexible: %v", err, err2, err3)
			}
			// Convert all fields to strings
			structured.Who = extractStringValue(flexible["who"])
			structured.What = extractStringValue(flexible["what"])
			structured.Why = extractStringValue(flexible["why"])
			structured.When = extractStringValue(flexible["when"])
			structured.Where = extractStringValue(flexible["where"])
			structured.How = extractStringValue(flexible["how"])
		} else {
			if len(arr) == 0 {
				return fmt.Errorf("LLM returned empty array")
			}
			structured = arr[0]
		}
	}

	// Create searchable content combining all fields
	searchContent := fmt.Sprintf("%s %s %s %s %s %s",
		structured.Who, structured.What, structured.Why,
		structured.When, structured.Where, structured.How)

	embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
	if err != nil {
		return err
	}

	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     doc.Content,
		ContentType: ContentTypeFact,
		Strategy:    StrategyWhoWhatWhy,
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:   doc.Content,
			Who:            structured.Who,
			What:           structured.What,
			Why:            structured.Why,
			When:           structured.When,
			Where:          structured.Where,
			How:            structured.How,
			SearchKeywords: strings.Fields(searchContent),
			SourceDocument: doc.RelativePath,
			DocumentType:   string(doc.Type),
			DocumentHash:   doc.Hash,
			Timestamp:      doc.ImportedAt,
		},
	}

	di.vectorDB.AddChunk(chunk)
	return nil
}

// processKeyword creates keyword-based chunks
func (di *DocumentImporter) processKeyword(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting keywords and key phrases"
	}

	prompt := fmt.Sprintf(`Extract the most important keywords and key phrases from this text.
Return ONLY a JSON array of strings: ["keyword1", "keyword2", ...]

Text:
%s`, doc.Content[:min(2000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no keywords found")
	}

	var keywords []string
	if err := json.Unmarshal([]byte(jsonStr), &keywords); err != nil {
		return err
	}

	// Create a chunk with keyword metadata
	embedding, err := di.client.GenerateEmbedding(embedModel, strings.Join(keywords, " "))
	if err != nil {
		return err
	}

	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     doc.Content,
		ContentType: ContentTypeFact,
		Strategy:    StrategyKeyword,
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:   doc.Content,
			SearchKeywords: keywords,
			FactKeywords:   keywords,
			SourceDocument: doc.RelativePath,
			DocumentType:   string(doc.Type),
			DocumentHash:   doc.Hash,
			Timestamp:      doc.ImportedAt,
		},
	}

	di.vectorDB.AddChunk(chunk)
	return nil
}

// processSentence creates sentence-level chunks
func (di *DocumentImporter) processSentence(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Creating sentence-level chunks"
	}

	// Simple sentence splitting
	sentences := strings.Split(doc.Content, ".")

	for i, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) < 10 {
			continue
		}

		embedding, err := di.client.GenerateEmbedding(embedModel, sentence)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     sentence,
			ContentType: ContentTypeFact,
			Strategy:    StrategySentence,
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				SentenceIndex:  i,
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// processFullQA creates full Q&A pair chunks
func (di *DocumentImporter) processFullQA(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Generating Q&A pairs"
	}

	prompt := fmt.Sprintf(`Generate question-answer pairs from this text.
For each important piece of information, create a natural question and its answer.

Text:
%s

Return ONLY a JSON array:
[{"question": "question text", "answer": "answer text"}]`, doc.Content[:min(2000, len(doc.Content))])

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true)
	if jsonStr == "" {
		return fmt.Errorf("no Q&A pairs found")
	}

	var pairs []struct {
		Question string `json:"question"`
		Answer   string `json:"answer"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &pairs); err != nil {
		return err
	}

	for _, pair := range pairs {
		embedding, err := di.client.GenerateEmbedding(embedModel, pair.Question)
		if err != nil {
			continue
		}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     pair.Answer,
			ContentType: ContentTypeFact,
			Strategy:    StrategyFullQA,
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				SourceDocument: doc.RelativePath,
				DocumentType:   string(doc.Type),
				DocumentHash:   doc.Hash,
				Timestamp:      doc.ImportedAt,
			},
		}
		chunk.CanonicalQuestions = []string{pair.Question}
		chunk.CanonicalAnswer = pair.Answer

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

