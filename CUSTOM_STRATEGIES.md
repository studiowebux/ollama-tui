# Custom Import Strategies Guide

## Overview

Adding custom import strategies is straightforward - just add a new case to the switch statement and implement the processing function.

## Built-in Strategies

- `auto` - Auto-detect based on content type (default)
- `all` - Apply all strategies for maximum retrieval coverage
- `entity_sheet` - Extract characters, locations, items (fictional content)
- `who_what_why` - Structured Q&A extraction (factual content)
- `keyword` - Keyword-based chunking
- `sentence` - Sentence-level granularity
- `full_qa` - Generate complete Q&A pairs
- `document_section` - Markdown section splitting
- `code_snippet` - Code extraction with summaries

## How to Add a Custom Strategy

### Step 1: Add to Switch Statement

Edit `document_import_strategies.go` and add your strategy:

```go
func (di *DocumentImporter) ProcessWithStrategy(doc ImportedDocument, strategy string, chatModel, embedModel string, progressChan chan<- string) error {
	switch strategy {
	// ... existing cases ...
	case "my_custom_strategy":
		return di.processMyCustomStrategy(doc, chatModel, embedModel, progressChan)
	default:
		return di.processAuto(doc, chatModel, embedModel, progressChan)
	}
}
```

### Step 2: Implement Processing Function

Add your processing logic:

```go
func (di *DocumentImporter) processMyCustomStrategy(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Running my custom strategy"
	}

	// 1. Extract information using LLM
	prompt := fmt.Sprintf(`Your custom extraction prompt here.

	Text:
	%s

	Return ONLY JSON: {...}`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	// 2. Parse LLM response
	jsonStr := extractJSON(response, false)
	var extracted YourCustomStruct
	json.Unmarshal([]byte(jsonStr), &extracted)

	// 3. Create chunks with embeddings
	embedding, err := di.client.GenerateEmbedding(embedModel, extracted.Content)
	if err != nil {
		return err
	}

	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     extracted.Content,
		ContentType: ContentTypeFact, // or ContentTypeFictional
		Strategy:    "my_custom_strategy",
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:   doc.Content,
			SearchKeywords: extracted.Keywords,
			SourceDocument: doc.RelativePath,
			DocumentType:   string(doc.Type),
			DocumentHash:   doc.Hash,
			Timestamp:      doc.ImportedAt,
		},
	}

	// 4. Save chunk
	di.vectorDB.AddChunk(chunk)
	return nil
}
```

### Step 3: Add to Auto-completion

Edit `import_cli.go`:

```go
func completeStrategies(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	strategies := []string{
		// ... existing strategies ...
		"my_custom_strategy\tMy custom processing description",
	}
	// ...
}
```

### Step 4: Rebuild and Use

```bash
go build -o ollamatui .
./ollamatui import /path/to/docs --strategy my_custom_strategy --verbose
```

## Strategy Template

Use this template for any new strategy:

```go
func (di *DocumentImporter) processYourStrategy(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Processing with your strategy"
	}

	// Step 1: Define extraction prompt
	prompt := fmt.Sprintf(`Extract specific information.

	Text: %s

	Return JSON: {...}`, doc.Content[:min(2000, len(doc.Content))])

	// Step 2: Call LLM
	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	// Step 3: Parse response
	jsonStr := extractJSON(response, false) // false for single object, true for array
	if jsonStr == "" {
		return fmt.Errorf("no data extracted")
	}

	var data struct {
		Field1 string   `json:"field1"`
		Field2 []string `json:"field2"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return err
	}

	// Step 4: Create embedding
	searchContent := data.Field1 + " " + strings.Join(data.Field2, " ")
	embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
	if err != nil {
		return err
	}

	// Step 5: Build chunk
	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     doc.Content,
		ContentType: ContentTypeFact,
		Strategy:    "your_strategy",
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:   doc.Content,
			SearchKeywords: data.Field2,
			SourceDocument: doc.RelativePath,
			DocumentType:   string(doc.Type),
			DocumentHash:   doc.Hash,
			Timestamp:      doc.ImportedAt,
			// Add custom metadata fields here
		},
	}

	// Step 6: Save
	return di.vectorDB.AddChunk(chunk)
}
```

## Key Metadata Fields

Populate these for better retrieval:

- `SearchKeywords` - Keywords to boost in hybrid search
- `Entities` - Named entities (people, places, things)
- `CharacterRefs` - Character names (fictional content)
- `LocationRefs` - Location names (fictional content)
- `FactKeywords` - Factual keywords
- `EntityKey` / `EntityValue` - For entity lookup
- `Who/What/Why/When/Where/How` - Structured Q&A
- `CanonicalQuestions` - Natural language questions this chunk answers
- `CanonicalAnswer` - The answer to those questions

## Example: Timeline Strategy

```go
func (di *DocumentImporter) processTimeline(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	prompt := fmt.Sprintf(`Extract timeline events from this text.
	Each event should have: date/time, event description, participants.

	Text: %s

	Return JSON array: [{"when": "...", "what": "...", "who": "..."}]`, doc.Content)

	messages := []ChatMessage{{Role: "user", Content: prompt}}
	response, err := di.client.Chat(chatModel, messages)
	if err != nil {
		return err
	}

	jsonStr := extractJSON(response, true) // true for array
	var events []struct {
		When string `json:"when"`
		What string `json:"what"`
		Who  string `json:"who"`
	}
	json.Unmarshal([]byte(jsonStr), &events)

	for _, event := range events {
		searchContent := fmt.Sprintf("%s %s %s", event.When, event.What, event.Who)
		embedding, _ := di.client.GenerateEmbedding(embedModel, searchContent)

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     event.What,
			ContentType: ContentTypeFact,
			Strategy:    "timeline",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:   doc.Content,
				When:           event.When,
				What:           event.What,
				Who:            event.Who,
				SearchKeywords: []string{event.When, event.Who},
				SourceDocument: doc.RelativePath,
				DocumentHash:   doc.Hash,
			},
		}
		di.vectorDB.AddChunk(chunk)
	}
	return nil
}
```

## That's It!

Adding a strategy is just:
1. Add case to switch (1 line)
2. Implement processing function (~50 lines)
3. Add to auto-completion (1 line)
4. Rebuild

No complex framework, no configuration files, just Go code.
