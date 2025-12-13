package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ContentType defines how content should be chunked and retrieved
type ContentType string

const (
	ContentTypeFact       ContentType = "fact"        // Factual Q&A, 1:1 mapping
	ContentTypeFictional  ContentType = "fictional"   // Stories, rules, world-building
	ContentTypeCode       ContentType = "code"        // Technical docs, code examples
	ContentTypeDialog     ContentType = "dialog"      // Conversational, context-heavy
)

// ChunkStrategy defines the indexing strategy used
type ChunkStrategy string

const (
	StrategyFullQA       ChunkStrategy = "full_qa"        // Complete Q&A pair
	StrategySentence     ChunkStrategy = "sentence"       // Individual sentence
	StrategyKeyValue     ChunkStrategy = "key_value"      // Entity: Description
	StrategyWhoWhatWhy   ChunkStrategy = "who_what_why"   // Structured Q&A
	StrategyKeyword      ChunkStrategy = "keyword"        // Keyword-based
	StrategyEntitySheet  ChunkStrategy = "entity_sheet"   // Character/location sheet
	StrategyQuestionKey  ChunkStrategy = "question_key"   // Generated question as key, content as answer
)

// StoredContent represents deduplicated content
type StoredContent struct {
	Hash      string    `json:"hash"`
	Content   string    `json:"content"`
	RefCount  int       `json:"ref_count"`
	CreatedAt time.Time `json:"created_at"`
}

// VectorChunk represents a single embedded chunk of conversation
type VectorChunk struct {
	ID          string        `json:"id"`
	ChatID      string        `json:"chat_id"`
	ContentHash string        `json:"content_hash"` // Reference to StoredContent
	Content     string        `json:"content"`      // Kept for backward compatibility
	ContentType ContentType   `json:"content_type"`
	Strategy    ChunkStrategy `json:"strategy"`
	Embedding   []float64     `json:"embedding"`
	Metadata    ChunkMetadata `json:"metadata"`
	CreatedAt   time.Time     `json:"created_at"`

	// Canonical Q&A pairs for better matching
	CanonicalQuestions []string `json:"canonical_questions"`
	CanonicalAnswer    string   `json:"canonical_answer"`
}

// ChunkMetadata stores additional context about the chunk
type ChunkMetadata struct {
	UserMessage      string    `json:"user_message"`
	AssistantMessage string    `json:"assistant_message"`
	Timestamp        time.Time `json:"timestamp"`
	MarkedBad        bool      `json:"marked_bad"`
	Verified         bool      `json:"verified"`
	Tags             []string  `json:"tags"`
	Entities         []string  `json:"entities"`
	Topics           []string  `json:"topics"`
	ParentChunkID    string    `json:"parent_chunk_id"`
	RelatedChunkIDs  []string  `json:"related_chunk_ids"`
	ContextWindow    int       `json:"context_window"`

	// Fact-specific fields
	FactStatement string   `json:"fact_statement"`
	FactSource    string   `json:"fact_source"`
	FactKeywords  []string `json:"fact_keywords"`

	// Fictional content fields
	WorldElement   string   `json:"world_element"`
	RuleSystem     string   `json:"rule_system"`
	CharacterRefs  []string `json:"character_refs"`
	LocationRefs   []string `json:"location_refs"`
	SearchKeywords []string `json:"search_keywords"`

	// Key-Value indexing
	EntityKey   string `json:"entity_key"`   // e.g., "The Beggar of Somewhere"
	EntityValue string `json:"entity_value"` // e.g., full character sheet

	// Structured Q&A
	Who  string `json:"who"`  // Who is involved
	What string `json:"what"` // What happens/is described
	Why  string `json:"why"`  // Why it matters
	When string `json:"when"` // Temporal context
	Where string `json:"where"` // Spatial context
	How   string `json:"how"`  // How it works

	// Sentence-level granularity
	SentenceIndex int    `json:"sentence_index"` // Position in original text
	OriginalText  string `json:"original_text"`  // Full original message
}

// ContentStore manages deduplicated content
type ContentStore struct {
	contents map[string]*StoredContent
	dataDir  string
}

// VectorDB manages the vector database
type VectorDB struct {
	dataDir        string
	chunks         []VectorChunk
	projectManager *ProjectManager
	currentProject string
	contentStore   *ContentStore
}

// SearchResult represents a similarity search result
type SearchResult struct {
	Chunk      VectorChunk
	Similarity float64
}

func NewContentStore(dataDir string) (*ContentStore, error) {
	storeDir := filepath.Join(dataDir, "content")
	if err := os.MkdirAll(storeDir, 0755); err != nil {
		return nil, err
	}

	cs := &ContentStore{
		contents: make(map[string]*StoredContent),
		dataDir:  storeDir,
	}

	if err := cs.loadAll(); err != nil {
		return nil, err
	}

	return cs, nil
}

func (cs *ContentStore) hashContent(content string) string {
	hash := sha256.Sum256([]byte(content))
	return hex.EncodeToString(hash[:])
}

func (cs *ContentStore) Store(content string) (string, error) {
	hash := cs.hashContent(content)

	if existing, ok := cs.contents[hash]; ok {
		existing.RefCount++
		return hash, cs.save(existing)
	}

	stored := &StoredContent{
		Hash:      hash,
		Content:   content,
		RefCount:  1,
		CreatedAt: time.Now(),
	}

	cs.contents[hash] = stored
	return hash, cs.save(stored)
}

func (cs *ContentStore) Get(hash string) (string, bool) {
	if content, ok := cs.contents[hash]; ok {
		return content.Content, true
	}
	return "", false
}

func (cs *ContentStore) DecrementRef(hash string) error {
	if content, ok := cs.contents[hash]; ok {
		content.RefCount--
		if content.RefCount <= 0 {
			delete(cs.contents, hash)
			path := filepath.Join(cs.dataDir, hash+".json")
			return os.Remove(path)
		}
		return cs.save(content)
	}
	return nil
}

func (cs *ContentStore) save(content *StoredContent) error {
	data, err := json.MarshalIndent(content, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(cs.dataDir, content.Hash+".json")
	return os.WriteFile(path, data, 0644)
}

func (cs *ContentStore) loadAll() error {
	files, err := os.ReadDir(cs.dataDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) != ".json" {
			continue
		}

		path := filepath.Join(cs.dataDir, file.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		var content StoredContent
		if err := json.Unmarshal(data, &content); err != nil {
			continue
		}

		cs.contents[content.Hash] = &content
	}

	return nil
}

func NewVectorDB(pm *ProjectManager, projectID string) (*VectorDB, error) {
	dataDir := pm.GetVectorsPath(projectID)
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}

	contentStore, err := NewContentStore(dataDir)
	if err != nil {
		return nil, err
	}

	db := &VectorDB{
		dataDir:        dataDir,
		chunks:         []VectorChunk{},
		projectManager: pm,
		currentProject: projectID,
		contentStore:   contentStore,
	}

	if err := db.loadAllChunks(); err != nil {
		return nil, err
	}

	return db, nil
}

func (db *VectorDB) SwitchProject(projectID string) error {
	dataDir := db.projectManager.GetVectorsPath(projectID)
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return err
	}

	contentStore, err := NewContentStore(dataDir)
	if err != nil {
		return err
	}

	db.dataDir = dataDir
	db.currentProject = projectID
	db.contentStore = contentStore
	db.chunks = []VectorChunk{}
	return db.loadAllChunks()
}

// AddChunk stores a new vector chunk with content deduplication
func (db *VectorDB) AddChunk(chunk VectorChunk) error {
	chunk.ID = uuid.New().String()
	chunk.CreatedAt = time.Now()

	// Store content in content store and get hash
	if chunk.Content != "" {
		hash, err := db.contentStore.Store(chunk.Content)
		if err != nil {
			return err
		}
		chunk.ContentHash = hash
	}

	db.chunks = append(db.chunks, chunk)

	return db.saveChunk(chunk)
}

// GetChunkContent retrieves the actual content for a chunk
func (db *VectorDB) GetChunkContent(chunk *VectorChunk) string {
	// Try to get from content store first
	if chunk.ContentHash != "" {
		if content, ok := db.contentStore.Get(chunk.ContentHash); ok {
			return content
		}
	}
	// Fallback to embedded content for backward compatibility
	return chunk.Content
}

// Search finds the most similar chunks to the query embedding
// Excludes chunks marked as bad by default
func (db *VectorDB) Search(queryEmbedding []float64, topK int) []SearchResult {
	results := make([]SearchResult, 0, len(db.chunks))

	for _, chunk := range db.chunks {
		// Skip chunks marked as bad
		if chunk.Metadata.MarkedBad {
			continue
		}

		similarity := cosineSimilarity(queryEmbedding, chunk.Embedding)
		results = append(results, SearchResult{
			Chunk:      chunk,
			Similarity: similarity,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return results
}

// DeleteChatChunks removes all chunks associated with a chat
func (db *VectorDB) DeleteChatChunks(chatID string) error {
	filtered := make([]VectorChunk, 0)
	for _, chunk := range db.chunks {
		if chunk.ChatID != chatID {
			filtered = append(filtered, chunk)
		} else {
			// Decrement content reference count
			if chunk.ContentHash != "" {
				db.contentStore.DecrementRef(chunk.ContentHash)
			}
			// Delete the chunk file
			path := filepath.Join(db.dataDir, chunk.ID+".json")
			os.Remove(path)
		}
	}
	db.chunks = filtered
	return nil
}

func (db *VectorDB) saveChunk(chunk VectorChunk) error {
	data, err := json.MarshalIndent(chunk, "", "  ")
	if err != nil {
		return err
	}

	path := filepath.Join(db.dataDir, chunk.ID+".json")
	return os.WriteFile(path, data, 0644)
}

func (db *VectorDB) loadAllChunks() error {
	files, err := os.ReadDir(db.dataDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) != ".json" {
			continue
		}

		path := filepath.Join(db.dataDir, file.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		var chunk VectorChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}

		db.chunks = append(db.chunks, chunk)
	}

	return nil
}

// MarkChunkBad marks a chunk as containing bad/invalid information
func (db *VectorDB) MarkChunkBad(chunkID string) error {
	for i, chunk := range db.chunks {
		if chunk.ID == chunkID {
			db.chunks[i].Metadata.MarkedBad = true
			return db.saveChunk(db.chunks[i])
		}
	}
	return nil
}

// DeleteChunk permanently removes a chunk
func (db *VectorDB) DeleteChunk(chunkID string) error {
	filtered := make([]VectorChunk, 0)
	for _, chunk := range db.chunks {
		if chunk.ID != chunkID {
			filtered = append(filtered, chunk)
		} else {
			// Decrement content reference count
			if chunk.ContentHash != "" {
				db.contentStore.DecrementRef(chunk.ContentHash)
			}
			path := filepath.Join(db.dataDir, chunk.ID+".json")
			os.Remove(path)
		}
	}
	db.chunks = filtered
	return nil
}

// GetStats returns statistics about the vector database
func (db *VectorDB) GetStats() map[string]interface{} {
	totalChunks := len(db.chunks)
	markedBad := 0
	verified := 0
	chatIDs := make(map[string]bool)

	for _, chunk := range db.chunks {
		if chunk.Metadata.MarkedBad {
			markedBad++
		}
		if chunk.Metadata.Verified {
			verified++
		}
		chatIDs[chunk.ChatID] = true
	}

	return map[string]interface{}{
		"total_chunks":    totalChunks,
		"marked_bad":      markedBad,
		"verified":        verified,
		"unique_chats":    len(chatIDs),
		"storage_path":    db.dataDir,
		"stored_contents": len(db.contentStore.contents),
	}
}

// GetAllChunks returns all chunks for management
func (db *VectorDB) GetAllChunks() []VectorChunk {
	return db.chunks
}

// ClearAll deletes all vector chunks
func (db *VectorDB) ClearAll() error {
	files, err := os.ReadDir(db.dataDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			path := filepath.Join(db.dataDir, file.Name())
			if err := os.Remove(path); err != nil {
				return err
			}
		}
	}

	// Clear content store
	contentFiles, err := os.ReadDir(db.contentStore.dataDir)
	if err == nil {
		for _, file := range contentFiles {
			if filepath.Ext(file.Name()) == ".json" {
				path := filepath.Join(db.contentStore.dataDir, file.Name())
				os.Remove(path)
			}
		}
	}
	db.contentStore.contents = make(map[string]*StoredContent)

	db.chunks = []VectorChunk{}
	return nil
}

// GetChunkByID retrieves a specific chunk
func (db *VectorDB) GetChunkByID(id string) *VectorChunk {
	for _, chunk := range db.chunks {
		if chunk.ID == id {
			return &chunk
		}
	}
	return nil
}

// SearchWithContext expands results with related chunks
func (db *VectorDB) SearchWithContext(queryEmbedding []float64, topK int, includeRelated bool) []SearchResult {
	results := db.Search(queryEmbedding, topK)

	if !includeRelated {
		return results
	}

	// Expand with related chunks
	expanded := make(map[string]SearchResult)
	for _, result := range results {
		expanded[result.Chunk.ID] = result

		// Add parent context
		if result.Chunk.Metadata.ParentChunkID != "" {
			parent := db.GetChunkByID(result.Chunk.Metadata.ParentChunkID)
			if parent != nil {
				expanded[parent.ID] = SearchResult{
					Chunk:      *parent,
					Similarity: result.Similarity * 0.9, // Slightly lower score
				}
			}
		}

		// Add related chunks
		for _, relatedID := range result.Chunk.Metadata.RelatedChunkIDs {
			related := db.GetChunkByID(relatedID)
			if related != nil {
				expanded[related.ID] = SearchResult{
					Chunk:      *related,
					Similarity: result.Similarity * 0.85, // Lower score
				}
			}
		}
	}

	// Convert back to slice and sort
	expandedResults := make([]SearchResult, 0, len(expanded))
	for _, result := range expanded {
		expandedResults = append(expandedResults, result)
	}

	sort.Slice(expandedResults, func(i, j int) bool {
		return expandedResults[i].Similarity > expandedResults[j].Similarity
	})

	return expandedResults
}

// FindByEntity searches chunks by entity name
func (db *VectorDB) FindByEntity(entity string) []VectorChunk {
	results := make([]VectorChunk, 0)
	entityLower := strings.ToLower(entity)

	for _, chunk := range db.chunks {
		if chunk.Metadata.MarkedBad {
			continue
		}
		for _, e := range chunk.Metadata.Entities {
			if strings.ToLower(e) == entityLower {
				results = append(results, chunk)
				break
			}
		}
	}

	return results
}

// FindByTopic searches chunks by topic
func (db *VectorDB) FindByTopic(topic string) []VectorChunk {
	results := make([]VectorChunk, 0)
	topicLower := strings.ToLower(topic)

	for _, chunk := range db.chunks {
		if chunk.Metadata.MarkedBad {
			continue
		}
		for _, t := range chunk.Metadata.Topics {
			if strings.ToLower(t) == topicLower {
				results = append(results, chunk)
				break
			}
		}
	}

	return results
}

// SearchHybrid combines semantic similarity with keyword matching for better recall
func (db *VectorDB) SearchHybrid(queryEmbedding []float64, queryText string, topK int) []SearchResult {
	results := make([]SearchResult, 0, len(db.chunks))
	queryLower := strings.ToLower(queryText)
	queryWords := strings.Fields(queryLower)

	for _, chunk := range db.chunks {
		if chunk.Metadata.MarkedBad {
			continue
		}

		// Calculate semantic similarity
		semanticScore := cosineSimilarity(queryEmbedding, chunk.Embedding)

		// Calculate keyword match boost
		keywordBoost := 0.0

		// Check search keywords (fictional content)
		if len(chunk.Metadata.SearchKeywords) > 0 {
			for _, keyword := range chunk.Metadata.SearchKeywords {
				keywordLower := strings.ToLower(keyword)
				for _, queryWord := range queryWords {
					if strings.Contains(keywordLower, queryWord) || strings.Contains(queryWord, keywordLower) {
						keywordBoost += 0.15
					}
				}
			}
		}

		// Check character references
		if len(chunk.Metadata.CharacterRefs) > 0 {
			for _, char := range chunk.Metadata.CharacterRefs {
				charLower := strings.ToLower(char)
				for _, queryWord := range queryWords {
					if strings.Contains(charLower, queryWord) || strings.Contains(queryWord, charLower) {
						keywordBoost += 0.20
					}
				}
			}
		}

		// Check location references
		if len(chunk.Metadata.LocationRefs) > 0 {
			for _, loc := range chunk.Metadata.LocationRefs {
				locLower := strings.ToLower(loc)
				for _, queryWord := range queryWords {
					if strings.Contains(locLower, queryWord) || strings.Contains(queryWord, locLower) {
						keywordBoost += 0.15
					}
				}
			}
		}

		// Check entities
		if len(chunk.Metadata.Entities) > 0 {
			for _, entity := range chunk.Metadata.Entities {
				entityLower := strings.ToLower(entity)
				for _, queryWord := range queryWords {
					if strings.Contains(entityLower, queryWord) || strings.Contains(queryWord, entityLower) {
						keywordBoost += 0.10
					}
				}
			}
		}

		// Check fact keywords
		if len(chunk.Metadata.FactKeywords) > 0 {
			for _, keyword := range chunk.Metadata.FactKeywords {
				keywordLower := strings.ToLower(keyword)
				for _, queryWord := range queryWords {
					if strings.Contains(keywordLower, queryWord) || strings.Contains(queryWord, keywordLower) {
						keywordBoost += 0.10
					}
				}
			}
		}

		// Check entity key (strongest boost for exact entity lookups)
		if chunk.Metadata.EntityKey != "" {
			entityKeyLower := strings.ToLower(chunk.Metadata.EntityKey)
			for _, queryWord := range queryWords {
				if strings.Contains(entityKeyLower, queryWord) || strings.Contains(queryWord, entityKeyLower) {
					keywordBoost += 0.25
				}
			}
		}

		// Check canonical questions (VERY strong boost for exact matches)
		if len(chunk.CanonicalQuestions) > 0 {
			for _, canonQ := range chunk.CanonicalQuestions {
				canonLower := strings.ToLower(canonQ)
				// Exact match or high similarity
				if strings.Contains(canonLower, queryLower) || strings.Contains(queryLower, canonLower) {
					keywordBoost += 0.30
				}
				// Word-by-word matching
				for _, queryWord := range queryWords {
					if strings.Contains(canonLower, queryWord) {
						keywordBoost += 0.05
					}
				}
			}
		}

		// Cap keyword boost at 0.45 to prevent overwhelming semantic score
		if keywordBoost > 0.45 {
			keywordBoost = 0.45
		}

		// Combine scores: 70% semantic + 30% keyword (when keyword matches exist)
		finalScore := semanticScore + keywordBoost

		results = append(results, SearchResult{
			Chunk:      chunk,
			Similarity: finalScore,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return results
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
