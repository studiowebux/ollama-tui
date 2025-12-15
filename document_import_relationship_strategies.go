package main

import (
	"fmt"
	"regexp"
	"strings"
)

// processTags extracts markdown tags (#tag) and creates searchable tag-based chunks
func (di *DocumentImporter) processTags(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting tags and categorization"
	}

	// Extract hashtags from markdown content
	tagPattern := regexp.MustCompile(`#([a-zA-Z0-9_-]+)`)
	matches := tagPattern.FindAllStringSubmatch(doc.Content, -1)

	if len(matches) == 0 {
		// No tags found, skip strategy
		return nil
	}

	// Collect unique tags
	tagMap := make(map[string]bool)
	for _, match := range matches {
		if len(match) > 1 {
			tagMap[match[1]] = true
		}
	}

	tags := make([]string, 0, len(tagMap))
	for tag := range tagMap {
		tags = append(tags, tag)
	}

	if len(tags) == 0 {
		return nil
	}

	// Create context around tags
	tagContext := fmt.Sprintf("Document %s contains topics: %s", doc.RelativePath, strings.Join(tags, ", "))

	embedding, err := di.client.GenerateEmbedding(embedModel, tagContext)
	if err != nil {
		return err
	}

	chunk := VectorChunk{
		ChatID:      "document_import",
		Content:     tagContext,
		ContentType: ContentTypeFact,
		Strategy:    "tags",
		Embedding:   embedding,
		Metadata: ChunkMetadata{
			OriginalText:     doc.Content,
			SearchKeywords:   tags,
			SourceDocument:   doc.RelativePath,
			DocumentType:     string(doc.Type),
			DocumentHash:     doc.Hash,
			Timestamp:        doc.ImportedAt,
			DocumentTags:     tags, // Store tags in metadata
		},
	}

	chunk.CanonicalQuestions = []string{
		fmt.Sprintf("What topics are covered in %s?", doc.RelativePath),
		fmt.Sprintf("What documents are tagged with %s?", strings.Join(tags, " or ")),
	}
	chunk.CanonicalAnswer = fmt.Sprintf("%s covers: %s", doc.RelativePath, strings.Join(tags, ", "))

	return di.vectorDB.AddChunk(chunk)
}

// processCrossReferences extracts links and references between documents
func (di *DocumentImporter) processCrossReferences(doc ImportedDocument, chatModel, embedModel string, progressChan chan<- string) error {
	if progressChan != nil {
		progressChan <- "Extracting document cross-references"
	}

	// Extract markdown links: [text](link)
	linkPattern := regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	matches := linkPattern.FindAllStringSubmatch(doc.Content, -1)

	// Extract wiki-style links: [[Document Name]]
	wikiPattern := regexp.MustCompile(`\[\[([^\]]+)\]\]`)
	wikiMatches := wikiPattern.FindAllStringSubmatch(doc.Content, -1)

	// Combine all references
	references := make([]struct {
		text string
		link string
	}, 0)

	for _, match := range matches {
		if len(match) >= 3 {
			text := match[1]
			link := match[2]
			// Only include internal references (relative paths, .md files, etc)
			if isInternalReference(link) {
				references = append(references, struct {
					text string
					link string
				}{text, link})
			}
		}
	}

	for _, match := range wikiMatches {
		if len(match) >= 2 {
			refDoc := match[1]
			references = append(references, struct {
				text string
				link string
			}{refDoc, refDoc})
		}
	}

	if len(references) == 0 {
		// No cross-references found
		return nil
	}

	// Create chunks for each reference to build knowledge graph
	for _, ref := range references {
		searchContent := fmt.Sprintf("%s references %s: %s", doc.RelativePath, ref.link, ref.text)

		embedding, err := di.client.GenerateEmbedding(embedModel, searchContent)
		if err != nil {
			continue
		}

		// Extract related document keywords
		keywords := []string{doc.RelativePath, ref.link, ref.text}

		chunk := VectorChunk{
			ChatID:      "document_import",
			Content:     searchContent,
			ContentType: ContentTypeFact,
			Strategy:    "cross_references",
			Embedding:   embedding,
			Metadata: ChunkMetadata{
				OriginalText:     doc.Content,
				SearchKeywords:   keywords,
				SourceDocument:   doc.RelativePath,
				DocumentType:     string(doc.Type),
				DocumentHash:     doc.Hash,
				Timestamp:        doc.ImportedAt,
				RelatedDocuments: []string{ref.link}, // Track related documents
			},
		}

		chunk.CanonicalQuestions = []string{
			fmt.Sprintf("What does %s reference?", doc.RelativePath),
			fmt.Sprintf("What documents reference %s?", ref.link),
			fmt.Sprintf("How are %s and %s related?", doc.RelativePath, ref.link),
		}
		chunk.CanonicalAnswer = fmt.Sprintf("%s links to %s with context: %s", doc.RelativePath, ref.link, ref.text)

		di.vectorDB.AddChunk(chunk)
	}

	return nil
}

// isInternalReference checks if a link is an internal document reference
func isInternalReference(link string) bool {
	link = strings.ToLower(link)

	// Skip external URLs
	if strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://") {
		// But allow localhost/local domains if needed
		if !strings.Contains(link, "localhost") && !strings.Contains(link, "127.0.0.1") {
			return false
		}
	}

	// Skip anchors without path
	if strings.HasPrefix(link, "#") {
		return false
	}

	// Include relative paths and .md files
	if strings.HasSuffix(link, ".md") || strings.HasPrefix(link, "./") || strings.HasPrefix(link, "../") {
		return true
	}

	// Include paths without extension (wiki-style)
	if !strings.Contains(link, "://") && !strings.Contains(link, "@") {
		return true
	}

	return false
}
