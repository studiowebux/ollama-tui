package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ImportDocumentWithStrategy imports a document using a specific chunking strategy
func (di *DocumentImporter) ImportDocumentWithStrategy(filePath, chatModel, embedModel, strategy string, force bool, progressChan chan<- string) error {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	if len(content) == 0 {
		return fmt.Errorf("file is empty")
	}

	info, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("failed to stat file: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	supportedExts := di.SupportedExtensions()
	docType, ok := supportedExts[ext]
	if !ok {
		docType = DocTypeOther
	}

	relPath, _ := filepath.Rel(di.basePath, filePath)
	contentStr := string(content)

	if len(strings.TrimSpace(contentStr)) < 10 {
		return fmt.Errorf("file content too short (< 10 chars)")
	}

	// Calculate hash
	hash := sha256.Sum256(content)
	hashStr := hex.EncodeToString(hash[:])

	// Check if this document hash already exists (unless force is enabled)
	if !force && di.vectorDB.HasDocumentHash(hashStr) {
		if progressChan != nil {
			progressChan <- fmt.Sprintf("Skipped (already imported): %s", relPath)
		}
		return fmt.Errorf("already imported")
	}

	doc := ImportedDocument{
		ID:           hashStr,
		FilePath:     filePath,
		RelativePath: relPath,
		Type:         docType,
		Content:      contentStr,
		Hash:         hashStr,
		ImportedAt:   time.Now(),
		LastModified: info.ModTime(),
	}

	// Track chunks before processing
	chunksBefore := len(di.vectorDB.GetAllChunks())

	// Use the specified strategy
	err = di.ProcessWithStrategy(doc, strategy, chatModel, embedModel, progressChan)

	// If processing failed, rollback any chunks that were added
	if err != nil {
		chunksAfter := len(di.vectorDB.GetAllChunks())
		if chunksAfter > chunksBefore {
			// Remove chunks that were added during failed processing
			di.vectorDB.RemoveChunksByDocumentHash(hashStr)
			if progressChan != nil {
				progressChan <- fmt.Sprintf("Rolled back %d chunks due to error", chunksAfter-chunksBefore)
			}
		}
	}

	return err
}
