package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
)

// ---------- CycloneDX-aligned BOM types ----------

// ModelBOM is the top-level CycloneDX ML-BOM document.
type ModelBOM struct {
	BOMFormat    string       `json:"bomFormat" yaml:"bomFormat"`
	SpecVersion  string       `json:"specVersion" yaml:"specVersion"`
	Version      int          `json:"version" yaml:"version"`
	SerialNumber string       `json:"serialNumber" yaml:"serialNumber"`
	Metadata     BOMMetadata  `json:"metadata" yaml:"metadata"`
	Components   []Component  `json:"components" yaml:"components"`
	Dependencies []Dependency `json:"dependencies,omitempty" yaml:"dependencies,omitempty"`
}

// Dependency records a CycloneDX dependency relationship.
type Dependency struct {
	Ref       string   `json:"ref" yaml:"ref"`
	DependsOn []string `json:"dependsOn,omitempty" yaml:"dependsOn,omitempty"`
}

// BOMMetadata describes when and how the BOM was generated.
type BOMMetadata struct {
	Timestamp string     `json:"timestamp" yaml:"timestamp"`
	Tools     []BOMTool  `json:"tools" yaml:"tools"`
	Component *Component `json:"component,omitempty" yaml:"component,omitempty"`
}

// BOMTool identifies the tool that generated the BOM.
type BOMTool struct {
	Vendor  string `json:"vendor" yaml:"vendor"`
	Name    string `json:"name" yaml:"name"`
	Version string `json:"version" yaml:"version"`
}

// Component is a CycloneDX component (model, tokenizer, adapter, etc.).
type Component struct {
	Type         string         `json:"type" yaml:"type"`
	BOMRef       string         `json:"bom-ref,omitempty" yaml:"bom-ref,omitempty"`
	Name         string         `json:"name" yaml:"name"`
	Version      string         `json:"version,omitempty" yaml:"version,omitempty"`
	Description  string         `json:"description,omitempty" yaml:"description,omitempty"`
	Hashes       []Hash         `json:"hashes,omitempty" yaml:"hashes,omitempty"`
	Licenses     []License      `json:"licenses,omitempty" yaml:"licenses,omitempty"`
	ExternalRefs []ExternalRef  `json:"externalReferences,omitempty" yaml:"externalReferences,omitempty"`
	Properties   []Property     `json:"properties,omitempty" yaml:"properties,omitempty"`
	ModelCard    *ModelCard     `json:"modelCard,omitempty" yaml:"modelCard,omitempty"`
	Evidence     []EvidenceItem `json:"evidence,omitempty" yaml:"evidence,omitempty"`
}

// Hash is a content hash (CycloneDX format).
type Hash struct {
	Algorithm string `json:"alg" yaml:"alg"`
	Content   string `json:"content" yaml:"content"`
}

// License for the model.
type License struct {
	ID   string `json:"id,omitempty" yaml:"id,omitempty"`
	Name string `json:"name,omitempty" yaml:"name,omitempty"`
	URL  string `json:"url,omitempty" yaml:"url,omitempty"`
}

// ExternalRef links to external resources.
type ExternalRef struct {
	Type string `json:"type" yaml:"type"`
	URL  string `json:"url" yaml:"url"`
}

// Property is a key-value extension point (SecAI trust metadata lives here).
type Property struct {
	Name  string `json:"name" yaml:"name"`
	Value string `json:"value" yaml:"value"`
}

// ModelCard captures ML-specific metadata.
type ModelCard struct {
	BaseModel       string        `json:"base_model,omitempty" yaml:"base_model,omitempty"`
	ModelFamily     string        `json:"model_family,omitempty" yaml:"model_family,omitempty"`
	Parameters      string        `json:"parameters,omitempty" yaml:"parameters,omitempty"`
	Quantization    *Quantization `json:"quantization,omitempty" yaml:"quantization,omitempty"`
	Tokenizer       *Tokenizer    `json:"tokenizer,omitempty" yaml:"tokenizer,omitempty"`
	Adapters        []Adapter     `json:"adapters,omitempty" yaml:"adapters,omitempty"`
	TrustLabels     []string      `json:"trust_labels,omitempty" yaml:"trust_labels,omitempty"`
	PromotionStatus string        `json:"promotion_status,omitempty" yaml:"promotion_status,omitempty"`
}

// Quantization records quantization lineage.
type Quantization struct {
	Method        string  `json:"method" yaml:"method"` // Q4_K_M, Q5_K_S, etc.
	BitsPerWeight float64 `json:"bits_per_weight,omitempty" yaml:"bits_per_weight,omitempty"`
	BaseModelHash string  `json:"base_model_hash,omitempty" yaml:"base_model_hash,omitempty"`
	QuantizedBy   string  `json:"quantized_by,omitempty" yaml:"quantized_by,omitempty"`
	QuantizedAt   string  `json:"quantized_at,omitempty" yaml:"quantized_at,omitempty"`
}

// Tokenizer describes the tokenizer associated with a model.
type Tokenizer struct {
	Name   string `json:"name,omitempty" yaml:"name,omitempty"`
	Type   string `json:"type,omitempty" yaml:"type,omitempty"` // BPE, SentencePiece, etc.
	Hash   string `json:"hash,omitempty" yaml:"hash,omitempty"`
	Source string `json:"source,omitempty" yaml:"source,omitempty"`
}

// Adapter represents a LoRA or other fine-tuning adapter.
type Adapter struct {
	Name      string `json:"name" yaml:"name"`
	Type      string `json:"type" yaml:"type"` // lora, qlora, etc.
	Rank      int    `json:"rank,omitempty" yaml:"rank,omitempty"`
	Hash      string `json:"hash,omitempty" yaml:"hash,omitempty"`
	BaseModel string `json:"base_model,omitempty" yaml:"base_model,omitempty"`
}

// ---------- BOM generation ----------

const bomToolVersion = "0.2.0"

// MaxFileSize is the maximum model file size to hash (10 GiB).
const MaxFileSize = 10 << 30

const maxScanEntries = 200_000

// GenerateBOM creates a BOM from a model directory or single file.
func GenerateBOM(path string, meta *ModelMetadata) (*ModelBOM, error) {
	info, err := os.Lstat(path) // Lstat to detect symlinks
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	// Reject symlinks, devices, FIFOs at the top level
	if !info.Mode().IsDir() && !info.Mode().IsRegular() {
		return nil, fmt.Errorf("refusing to scan non-regular file: %s (mode=%s)", path, info.Mode())
	}

	var components []Component
	if info.IsDir() {
		components, err = scanDirectory(path, info, meta)
		if err != nil {
			return nil, err
		}
	} else {
		comp, err := scanFile(path, meta)
		if err != nil {
			return nil, err
		}
		components = []Component{comp}
	}
	return assembleBOM(path, components, meta)
}

// GenerateBOMFromRoot inventories a target through an already-open os.Root.
// Daemon allowlist roots are opened once at startup, so replacing an ancestor
// after authorization cannot redirect hashing outside the approved tree.
func GenerateBOMFromRoot(root *os.Root, relativePath, displayPath string, meta *ModelMetadata) (*ModelBOM, error) {
	if root == nil || relativePath == "" || filepath.IsAbs(relativePath) || relativePath == ".." || strings.HasPrefix(relativePath, ".."+string(filepath.Separator)) {
		return nil, errors.New("invalid rooted model target")
	}
	info, err := root.Lstat(relativePath)
	if err != nil {
		return nil, fmt.Errorf("inspect rooted model target: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || (!info.IsDir() && !info.Mode().IsRegular()) {
		return nil, fmt.Errorf("refusing symlink or special rooted target %q", displayPath)
	}

	var components []Component
	if info.IsDir() {
		components, err = scanRootDirectory(root, relativePath, info, meta)
	} else {
		hash, hashedInfo, hashErr := hashRootRegularFileSHA256(root, relativePath)
		if hashErr != nil {
			return nil, hashErr
		}
		components = []Component{buildModelComponent(hashedInfo, hash, meta)}
	}
	if err != nil {
		return nil, err
	}
	return assembleBOM(displayPath, components, meta)
}

func assembleBOM(path string, components []Component, meta *ModelMetadata) (*ModelBOM, error) {
	bom := &ModelBOM{
		BOMFormat:    "CycloneDX",
		SpecVersion:  "1.6",
		Version:      1,
		SerialNumber: "urn:uuid:" + uuid.New().String(),
		Metadata: BOMMetadata{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Tools: []BOMTool{{
				Vendor:  "SecAI-Hub",
				Name:    "ai-model-bom",
				Version: bomToolVersion,
			}},
		},
		Components: components,
	}

	// Wire lineage into generation when metadata has quantization/adapter info
	if meta != nil {
		wireLineage(bom, meta)
	}

	// Build dependency graph for multi-component BOMs
	bom.Dependencies = buildDependencies(bom.Components)

	// Set primary component
	if len(bom.Components) > 0 {
		primary := bom.Components[0]
		bom.Metadata.Component = &primary
	}
	modelCount := 0
	for _, component := range bom.Components {
		if component.Type == "machine-learning-model" {
			modelCount++
		}
	}
	if modelCount == 0 {
		return nil, fmt.Errorf("no supported model files found in %q", path)
	}
	for _, issue := range ValidateBOM(bom) {
		if issue.Severity == "error" {
			return nil, fmt.Errorf("generated BOM is invalid: %s: %s", issue.Field, issue.Message)
		}
	}

	return bom, nil
}

// wireLineage automatically embeds lineage properties when metadata provides lineage info.
func wireLineage(bom *ModelBOM, meta *ModelMetadata) {
	for i, comp := range bom.Components {
		if comp.Type != "machine-learning-model" {
			continue
		}

		card := comp.ModelCard
		if card == nil {
			continue
		}

		outputHash := ""
		if len(comp.Hashes) > 0 {
			outputHash = comp.Hashes[0].Content
		}

		var chain *LineageChain

		// Build quantization lineage
		if card.Quantization != nil && card.Quantization.Method != "" && meta.BaseModel != "" {
			baseHash := meta.BaseModelHash
			chain = BuildQuantizationLineage(
				meta.BaseModel, baseHash,
				card.Quantization.Method, outputHash,
				meta.QuantizedBy,
			)
		}

		// Build adapter lineage
		if len(card.Adapters) > 0 && meta.BaseModel != "" {
			baseHash := meta.BaseModelHash
			chain = BuildAdapterLineage(meta.BaseModel, baseHash, card.Adapters, outputHash)
		}

		// Add promotion step
		if chain != nil && meta.PromotionStatus != "" {
			chain.AddPromotion(meta.PromotionStatus, "", meta.PromotionReason)
		}

		// If no chain built but we have a base model, create a simple one
		if chain == nil && meta.BaseModel != "" {
			chain = NewLineageChain(comp.Name, outputHash)
			chain.AddEntry("download", fmt.Sprintf("base model: %s", meta.BaseModel),
				"", meta.BaseModelHash, "", "", nil)
			if meta.PromotionStatus != "" {
				chain.AddPromotion(meta.PromotionStatus, "", meta.PromotionReason)
			}
		}

		// Embed lineage as properties
		if chain != nil {
			bom.Components[i].Properties = append(bom.Components[i].Properties, chain.ToProperties()...)
		}
	}
}

// buildDependencies creates CycloneDX dependency relationships.
func buildDependencies(components []Component) []Dependency {
	if len(components) < 2 {
		return nil
	}

	var deps []Dependency
	var modelRef string

	for _, comp := range components {
		if comp.Type == "machine-learning-model" && modelRef == "" {
			modelRef = comp.BOMRef
		}
	}

	if modelRef == "" {
		return nil
	}

	// Tokenizers and adapters depend on the primary model
	var dependsOn []string
	for _, comp := range components {
		if comp.BOMRef == modelRef {
			continue
		}
		for _, prop := range comp.Properties {
			if prop.Name == "secai:component_type" && (prop.Value == "tokenizer" || prop.Value == "adapter") {
				dependsOn = append(dependsOn, comp.BOMRef)
			}
		}
	}

	if len(dependsOn) > 0 {
		deps = append(deps, Dependency{Ref: modelRef, DependsOn: dependsOn})
	}

	return deps
}

// scanDirectory walks a directory and creates components for model files.
// Rejects symlinks, devices, FIFOs, and files exceeding MaxFileSize.
func scanDirectory(dir string, expectedRoot os.FileInfo, meta *ModelMetadata) ([]Component, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("open scan root: %w", err)
	}
	defer root.Close()
	openedRoot, err := root.Lstat(".")
	if err != nil || !os.SameFile(expectedRoot, openedRoot) || !openedRoot.IsDir() {
		return nil, fmt.Errorf("scan root changed while opening")
	}
	return scanRootDirectory(root, ".", openedRoot, meta)
}

func scanRootDirectory(root *os.Root, startDir string, expectedRoot os.FileInfo, meta *ModelMetadata) ([]Component, error) {
	var components []Component
	visitedEntries := 0
	startInfo, err := root.Lstat(startDir)
	if err != nil || !startInfo.IsDir() || !os.SameFile(expectedRoot, startInfo) {
		return nil, fmt.Errorf("rooted scan target changed before enumeration")
	}

	pending := []string{startDir}
	for len(pending) > 0 {
		relDir := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		expectedDir, err := root.Lstat(relDir)
		if err != nil || !expectedDir.IsDir() || expectedDir.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("scan directory %q changed or is not a real directory", relDir)
		}
		dirHandle, err := root.Open(relDir)
		if err != nil {
			return nil, fmt.Errorf("open scan directory %q: %w", relDir, err)
		}
		openedDir, err := dirHandle.Stat()
		if err != nil || !os.SameFile(expectedDir, openedDir) {
			_ = dirHandle.Close()
			return nil, fmt.Errorf("scan directory %q changed while opening", relDir)
		}

		for {
			entries, readErr := dirHandle.ReadDir(256)
			for _, entry := range entries {
				visitedEntries++
				if visitedEntries > maxScanEntries {
					_ = dirHandle.Close()
					return nil, fmt.Errorf("directory scan exceeds %d-entry limit", maxScanEntries)
				}
				relPath := filepath.Join(relDir, entry.Name())
				info, err := root.Lstat(relPath)
				if err != nil {
					_ = dirHandle.Close()
					return nil, fmt.Errorf("inspect %q: %w", relPath, err)
				}
				if info.Mode()&os.ModeSymlink != 0 || (!info.IsDir() && !info.Mode().IsRegular()) {
					_ = dirHandle.Close()
					return nil, fmt.Errorf("refusing symlink or special entry %q", relPath)
				}
				if info.IsDir() {
					pending = append(pending, relPath)
					continue
				}

				ext := strings.ToLower(filepath.Ext(info.Name()))
				isModel := ext == ".gguf" || ext == ".bin" || ext == ".safetensors" || ext == ".pt" || ext == ".onnx"
				isTokenizer := ext == ".json" && isTokenizerFile(info.Name())
				if !isModel && !isTokenizer {
					continue
				}
				hash, hashedInfo, err := hashRootRegularFileSHA256(root, relPath)
				if err != nil {
					_ = dirHandle.Close()
					return nil, err
				}
				if isModel {
					comp := buildModelComponent(hashedInfo, hash, meta)
					comp.BOMRef = filepath.ToSlash(relPath)
					components = append(components, comp)
				} else {
					comp := tokenizerComponent(hashedInfo, hash)
					comp.BOMRef = "tokenizer:" + filepath.ToSlash(relPath)
					components = append(components, comp)
				}
				if len(components) > 100000 {
					_ = dirHandle.Close()
					return nil, errors.New("component count exceeds 100000")
				}
			}
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				_ = dirHandle.Close()
				return nil, fmt.Errorf("enumerate scan directory %q: %w", relDir, readErr)
			}
		}
		if after, err := dirHandle.Stat(); err != nil || !os.SameFile(openedDir, after) ||
			!after.ModTime().Equal(openedDir.ModTime()) {
			_ = dirHandle.Close()
			return nil, fmt.Errorf("scan directory %q changed while enumerating", relDir)
		}
		if err := dirHandle.Close(); err != nil {
			return nil, err
		}
	}

	return components, nil
}

// scanFile creates a component from a single model file.
func scanFile(path string, meta *ModelMetadata) (Component, error) {
	hash, info, err := hashRegularFileSHA256(path)
	if err != nil {
		return Component{}, err
	}

	return buildModelComponent(info, hash, meta), nil
}

func buildModelComponent(info os.FileInfo, hash string, meta *ModelMetadata) Component {
	name := info.Name()
	comp := Component{
		Type:   "machine-learning-model",
		BOMRef: name,
		Name:   name,
		Hashes: []Hash{{Algorithm: "SHA-256", Content: hash}},
		Properties: []Property{
			{Name: "secai:file_size", Value: fmt.Sprintf("%d", info.Size())},
			{Name: "secai:file_format", Value: detectFormat(name)},
		},
	}

	// Extract metadata from filename
	parsed := parseModelFilename(name)
	if parsed.ModelFamily != "" || parsed.Quantization != "" {
		card := &ModelCard{
			ModelFamily: parsed.ModelFamily,
			Parameters:  parsed.Parameters,
		}
		if parsed.Quantization != "" {
			card.Quantization = &Quantization{
				Method: parsed.Quantization,
			}
		}
		comp.ModelCard = card
	}

	// Apply user-supplied metadata
	if meta != nil {
		applyMetadata(&comp, meta)
	}

	return comp
}

// buildTokenizerComponent creates a component for a tokenizer file.
func buildTokenizerComponent(path string, info os.FileInfo) (Component, error) {
	hash, hashedInfo, err := hashRegularFileSHA256(path)
	if err != nil {
		return Component{}, err
	}
	if !os.SameFile(info, hashedInfo) {
		return Component{}, fmt.Errorf("tokenizer file %q changed while being scanned", path)
	}
	return tokenizerComponent(hashedInfo, hash), nil
}

func tokenizerComponent(info os.FileInfo, hash string) Component {
	return Component{
		Type:   "data",
		BOMRef: "tokenizer:" + info.Name(),
		Name:   info.Name(),
		Hashes: []Hash{{Algorithm: "SHA-256", Content: hash}},
		Properties: []Property{
			{Name: "secai:component_type", Value: "tokenizer"},
		},
	}
}

// ---------- metadata types ----------

// ModelMetadata is optional user-supplied metadata for BOM enrichment.
type ModelMetadata struct {
	Name            string   `yaml:"name,omitempty"`
	Version         string   `yaml:"version,omitempty"`
	Description     string   `yaml:"description,omitempty"`
	BaseModel       string   `yaml:"base_model,omitempty"`
	BaseModelHash   string   `yaml:"base_model_hash,omitempty"`
	ModelFamily     string   `yaml:"model_family,omitempty"`
	License         string   `yaml:"license,omitempty"`
	SourceURL       string   `yaml:"source_url,omitempty"`
	TrustLabels     []string `yaml:"trust_labels,omitempty"`
	PromotionStatus string   `yaml:"promotion_status,omitempty"`
	PromotionReason string   `yaml:"promotion_reason,omitempty"`
	QuantizedBy     string   `yaml:"quantized_by,omitempty"`
	QuantizedAt     string   `yaml:"quantized_at,omitempty"`
}

// applyMetadata enriches a component with user-supplied metadata.
func applyMetadata(comp *Component, meta *ModelMetadata) {
	if meta.Name != "" {
		comp.Name = meta.Name
	}
	if meta.Version != "" {
		comp.Version = meta.Version
	}
	if meta.Description != "" {
		comp.Description = meta.Description
	}
	if meta.License != "" {
		comp.Licenses = []License{{Name: meta.License}}
	}
	if meta.SourceURL != "" {
		comp.ExternalRefs = append(comp.ExternalRefs, ExternalRef{
			Type: "distribution",
			URL:  meta.SourceURL,
		})
	}

	if comp.ModelCard == nil {
		comp.ModelCard = &ModelCard{}
	}
	if meta.BaseModel != "" {
		comp.ModelCard.BaseModel = meta.BaseModel
	}
	if meta.ModelFamily != "" {
		comp.ModelCard.ModelFamily = meta.ModelFamily
	}
	if meta.TrustLabels != nil {
		comp.ModelCard.TrustLabels = meta.TrustLabels
	}
	if meta.PromotionStatus != "" {
		comp.ModelCard.PromotionStatus = meta.PromotionStatus
	}
	if meta.QuantizedBy != "" && comp.ModelCard.Quantization != nil {
		comp.ModelCard.Quantization.QuantizedBy = meta.QuantizedBy
	}
	if meta.QuantizedAt != "" && comp.ModelCard.Quantization != nil {
		comp.ModelCard.Quantization.QuantizedAt = meta.QuantizedAt
	}
}

// ---------- filename parsing ----------

// parsedFilename holds metadata extracted from a model filename.
type parsedFilename struct {
	ModelFamily  string
	Parameters   string
	Quantization string
	Format       string
}

var (
	// Match patterns like "mistral-7b-instruct-v0.3.Q4_K_M.gguf"
	reQuant  = regexp.MustCompile(`\.(Q\d[_\w]*|F16|F32|BF16)\.`)
	reParams = regexp.MustCompile(`(?i)(\d+\.?\d*[bBmM])`)
)

// parseModelFilename extracts metadata from a model filename.
func parseModelFilename(name string) parsedFilename {
	p := parsedFilename{Format: detectFormat(name)}

	// Extract quantization
	if m := reQuant.FindStringSubmatch(name); len(m) > 1 {
		p.Quantization = m[1]
	}

	// Extract parameter count
	if m := reParams.FindStringSubmatch(name); len(m) > 1 {
		p.Parameters = strings.ToUpper(m[1])
	}

	// Extract model family (everything before parameter count or quantization)
	base := strings.TrimSuffix(name, filepath.Ext(name))
	// Remove quantization suffix
	if p.Quantization != "" {
		base = strings.Replace(base, "."+p.Quantization, "", 1)
		base = strings.Replace(base, "-"+p.Quantization, "", 1)
	}
	// Clean up
	base = strings.TrimRight(base, ".-_ ")
	if base != "" {
		p.ModelFamily = base
	}

	return p
}

// detectFormat returns the model file format.
func detectFormat(name string) string {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".gguf":
		return "GGUF"
	case ".safetensors":
		return "SafeTensors"
	case ".bin":
		return "PyTorch"
	case ".pt":
		return "PyTorch"
	case ".onnx":
		return "ONNX"
	default:
		return "unknown"
	}
}

func isTokenizerFile(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "tokenizer") || strings.Contains(lower, "vocab")
}

// ---------- helpers ----------

func hashRegularFileSHA256(path string) (string, os.FileInfo, error) {
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return "", nil, err
	}
	if !pathInfo.Mode().IsRegular() {
		return "", nil, fmt.Errorf("refusing non-regular file %q", path)
	}
	if pathInfo.Size() > MaxFileSize {
		return "", nil, fmt.Errorf("file %q exceeds %d-byte limit", path, MaxFileSize)
	}

	// #nosec G304 -- path is the caller-selected model input and is pinned to
	// the regular-file identity established by Lstat immediately above.
	f, err := os.Open(path)
	if err != nil {
		return "", nil, err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return "", nil, err
	}
	if !before.Mode().IsRegular() || !os.SameFile(pathInfo, before) ||
		pathInfo.Size() != before.Size() || !pathInfo.ModTime().Equal(before.ModTime()) {
		return "", nil, fmt.Errorf("file %q changed while being opened", path)
	}

	h := sha256.New()
	written, err := io.Copy(h, io.LimitReader(f, MaxFileSize+1))
	if err != nil {
		return "", nil, err
	}
	if written > MaxFileSize {
		return "", nil, fmt.Errorf("file %q exceeds %d-byte limit", path, MaxFileSize)
	}
	after, err := f.Stat()
	if err != nil {
		return "", nil, err
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return "", nil, fmt.Errorf("file %q changed while being hashed", path)
	}
	pathAfter, err := os.Lstat(path)
	if err != nil || !os.SameFile(after, pathAfter) || after.Size() != pathAfter.Size() ||
		!after.ModTime().Equal(pathAfter.ModTime()) {
		return "", nil, fmt.Errorf("file path %q changed while being hashed", path)
	}
	return hex.EncodeToString(h.Sum(nil)), after, nil
}

func hashRootRegularFileSHA256(root *os.Root, relativePath string) (string, os.FileInfo, error) {
	pathInfo, err := root.Lstat(relativePath)
	if err != nil {
		return "", nil, err
	}
	if !pathInfo.Mode().IsRegular() || pathInfo.Mode()&os.ModeSymlink != 0 {
		return "", nil, fmt.Errorf("refusing non-regular file %q", relativePath)
	}
	if pathInfo.Size() > MaxFileSize {
		return "", nil, fmt.Errorf("file %q exceeds %d-byte limit", relativePath, MaxFileSize)
	}

	f, err := root.Open(relativePath)
	if err != nil {
		return "", nil, err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !before.Mode().IsRegular() || !os.SameFile(pathInfo, before) ||
		pathInfo.Size() != before.Size() || !pathInfo.ModTime().Equal(before.ModTime()) {
		return "", nil, fmt.Errorf("file %q changed while being opened", relativePath)
	}

	h := sha256.New()
	written, err := io.Copy(h, io.LimitReader(f, MaxFileSize+1))
	if err != nil {
		return "", nil, err
	}
	if written > MaxFileSize {
		return "", nil, fmt.Errorf("file %q exceeds %d-byte limit", relativePath, MaxFileSize)
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() ||
		!before.ModTime().Equal(after.ModTime()) {
		return "", nil, fmt.Errorf("file %q changed while being hashed", relativePath)
	}
	pathAfter, err := root.Lstat(relativePath)
	if err != nil || !os.SameFile(after, pathAfter) || after.Size() != pathAfter.Size() ||
		!after.ModTime().Equal(pathAfter.ModTime()) {
		return "", nil, fmt.Errorf("file path %q changed while being hashed", relativePath)
	}
	return hex.EncodeToString(h.Sum(nil)), after, nil
}
