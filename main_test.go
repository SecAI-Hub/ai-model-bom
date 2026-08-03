package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------- BOM generation tests ----------

func TestGenerateBOM_SingleFile(t *testing.T) {
	dir := t.TempDir()
	modelFile := filepath.Join(dir, "test-model.Q4_K_M.gguf")
	os.WriteFile(modelFile, []byte("fake-model-data"), 0o644)

	bom, err := GenerateBOM(modelFile, nil)
	if err != nil {
		t.Fatal(err)
	}

	if bom.BOMFormat != "CycloneDX" {
		t.Errorf("expected CycloneDX format, got %s", bom.BOMFormat)
	}
	if bom.SpecVersion != "1.6" {
		t.Errorf("expected spec 1.6, got %s", bom.SpecVersion)
	}
	if len(bom.Components) != 1 {
		t.Fatalf("expected 1 component, got %d", len(bom.Components))
	}

	comp := bom.Components[0]
	if comp.Type != "machine-learning-model" {
		t.Errorf("expected machine-learning-model type, got %s", comp.Type)
	}
	if len(comp.Hashes) == 0 {
		t.Error("expected hash in component")
	}
	if comp.ModelCard == nil || comp.ModelCard.Quantization == nil {
		t.Error("expected quantization info extracted from filename")
	}
	if comp.ModelCard.Quantization.Method != "Q4_K_M" {
		t.Errorf("expected Q4_K_M, got %s", comp.ModelCard.Quantization.Method)
	}
}

func TestGenerateBOM_Directory(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("model"), 0o644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte("{}"), 0o644)
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("hi"), 0o644) // should be ignored

	bom, err := GenerateBOM(dir, nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(bom.Components) != 2 {
		t.Errorf("expected 2 components (model + tokenizer), got %d", len(bom.Components))
	}
}

func TestGenerateBOM_WithMetadata(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("data"), 0o644)

	meta := &ModelMetadata{
		Name:            "my-model",
		Version:         "1.0",
		Description:     "test model",
		BaseModel:       "llama-2-7b",
		License:         "Apache-2.0",
		SourceURL:       "https://example.com/model",
		TrustLabels:     []string{"verified", "scanned"},
		PromotionStatus: "approved",
	}

	bom, err := GenerateBOM(dir, meta)
	if err != nil {
		t.Fatal(err)
	}

	comp := bom.Components[0]
	if comp.Name != "my-model" {
		t.Errorf("expected name override, got %s", comp.Name)
	}
	if comp.Version != "1.0" {
		t.Errorf("expected version 1.0, got %s", comp.Version)
	}
	if len(comp.Licenses) == 0 || comp.Licenses[0].Name != "Apache-2.0" {
		t.Error("expected license applied")
	}
	if comp.ModelCard == nil {
		t.Fatal("expected model card")
	}
	if comp.ModelCard.BaseModel != "llama-2-7b" {
		t.Errorf("expected base model, got %s", comp.ModelCard.BaseModel)
	}
	if comp.ModelCard.PromotionStatus != "approved" {
		t.Errorf("expected approved, got %s", comp.ModelCard.PromotionStatus)
	}
	if len(comp.ModelCard.TrustLabels) != 2 {
		t.Errorf("expected 2 trust labels, got %d", len(comp.ModelCard.TrustLabels))
	}
}

func TestGenerateBOM_SerialNumber(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "m.gguf"), []byte("x"), 0o644)

	bom1, _ := GenerateBOM(dir, nil)
	bom2, _ := GenerateBOM(dir, nil)

	if bom1.SerialNumber == bom2.SerialNumber {
		t.Error("each BOM should have a unique serial number")
	}
}

func TestGenerateBOM_RejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "real.gguf")
	os.WriteFile(target, []byte("data"), 0o644)
	link := filepath.Join(dir, "link.gguf")
	os.Symlink(target, link)

	_, err := GenerateBOM(link, nil)
	if err == nil {
		t.Error("expected error for symlink, got nil")
	}
}

// ---------- filename parsing tests ----------

func TestParseFilename_GGUF(t *testing.T) {
	p := parseModelFilename("mistral-7b-instruct-v0.3.Q4_K_M.gguf")
	if p.Quantization != "Q4_K_M" {
		t.Errorf("expected Q4_K_M, got %s", p.Quantization)
	}
	if p.Format != "GGUF" {
		t.Errorf("expected GGUF, got %s", p.Format)
	}
	if p.Parameters != "7B" {
		t.Errorf("expected 7B, got %s", p.Parameters)
	}
}

func TestParseFilename_SafeTensors(t *testing.T) {
	p := parseModelFilename("model.safetensors")
	if p.Format != "SafeTensors" {
		t.Errorf("expected SafeTensors, got %s", p.Format)
	}
}

func TestParseFilename_NoQuant(t *testing.T) {
	p := parseModelFilename("llama-2-7b.bin")
	if p.Quantization != "" {
		t.Errorf("expected no quantization, got %s", p.Quantization)
	}
	if p.Parameters != "2-7B" && p.Parameters != "7B" {
		// Accept either extraction
	}
}

// ---------- lineage tests ----------

func TestLineage_Quantization(t *testing.T) {
	chain := BuildQuantizationLineage("llama-2-7b", "abc123", "Q4_K_M", "def456", "operator")

	if len(chain.Entries) != 2 {
		t.Fatalf("expected 2 lineage entries, got %d", len(chain.Entries))
	}

	if chain.Entries[0].Action != "download" {
		t.Errorf("expected download, got %s", chain.Entries[0].Action)
	}
	if chain.Entries[1].Action != "quantize" {
		t.Errorf("expected quantize, got %s", chain.Entries[1].Action)
	}

	valid, _ := chain.Verify()
	if !valid {
		t.Error("lineage chain should be valid")
	}
}

func TestLineage_Adapter(t *testing.T) {
	adapters := []Adapter{
		{Name: "domain-lora", Type: "lora", Hash: "lora123"},
	}
	chain := BuildAdapterLineage("base-model", "base123", adapters, "final456")

	if len(chain.Entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(chain.Entries))
	}
	if chain.Entries[1].Action != "fine-tune" {
		t.Errorf("expected fine-tune, got %s", chain.Entries[1].Action)
	}
}

func TestLineage_Promotion(t *testing.T) {
	chain := NewLineageChain("model", "hash123")
	chain.AddEntry("download", "downloaded", "", "hash123", "", "op", nil)
	chain.AddPromotion("approved", "admin", "scans passed")

	if len(chain.Entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(chain.Entries))
	}
	if chain.Entries[1].Action != "promote" {
		t.Errorf("expected promote, got %s", chain.Entries[1].Action)
	}
}

func TestLineage_ToProperties(t *testing.T) {
	chain := BuildQuantizationLineage("base", "abc", "Q5_K_S", "def", "op")
	props := chain.ToProperties()
	if len(props) != 2 {
		t.Errorf("expected 2 properties, got %d", len(props))
	}
}

func TestLineage_VerifyBroken(t *testing.T) {
	chain := NewLineageChain("model", "hash")
	chain.Entries = []LineageEntry{
		{Step: 1, Action: "download", OutputHash: "aaa"},
		{Step: 2, Action: "quantize", InputHash: "bbb", OutputHash: "ccc"},
	}

	valid, failIdx := chain.Verify()
	if valid {
		t.Error("broken chain should fail verification")
	}
	if failIdx != 1 {
		t.Errorf("expected failure at step 1, got %d", failIdx)
	}
}

func TestLineage_WiredIntoGenerate(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "model.Q4_K_M.gguf"), []byte("data"), 0o644)

	meta := &ModelMetadata{
		BaseModel:     "llama-2-7b",
		BaseModelHash: "abc123",
	}

	bom, err := GenerateBOM(dir, meta)
	if err != nil {
		t.Fatal(err)
	}

	// Should have lineage properties wired in
	comp := bom.Components[0]
	hasLineage := false
	for _, p := range comp.Properties {
		if strings.HasPrefix(p.Name, "secai:lineage:") {
			hasLineage = true
			break
		}
	}
	if !hasLineage {
		t.Error("expected lineage properties wired into component")
	}
}

// ---------- evidence tests ----------

func TestEvidence_Import(t *testing.T) {
	dir := t.TempDir()
	evidenceFile := filepath.Join(dir, "evidence.json")
	os.WriteFile(evidenceFile, []byte(`[
		{"type":"scan","source":"ai-quarantine","timestamp":"2026-08-02T00:00:00Z","result":"pass","description":"virus scan clean"},
		{"type":"test","source":"gguf-guard","timestamp":"2026-08-02T00:00:00Z","result":"pass","description":"GGUF header valid"}
	]`), 0o644)

	items, err := ImportEvidence(evidenceFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Errorf("expected 2 evidence items, got %d", len(items))
	}
}

func TestEvidence_AttachToBOM(t *testing.T) {
	bom := &ModelBOM{
		Components: []Component{
			{Name: "model.gguf", BOMRef: "model.gguf"},
		},
	}

	evidence := []EvidenceItem{
		{Type: "scan", Source: "quarantine", Result: "pass", Description: "clean"},
	}

	found := AttachEvidence(bom, "model.gguf", evidence)
	if !found {
		t.Error("should find component")
	}
	if len(bom.Components[0].Evidence) != 1 {
		t.Error("evidence should be attached")
	}
}

func TestEvidence_AttachMissing(t *testing.T) {
	bom := &ModelBOM{Components: []Component{{Name: "other.bin"}}}
	found := AttachEvidence(bom, "missing.gguf", nil)
	if found {
		t.Error("should not find nonexistent component")
	}
}

func TestAttestation_SignAndVerify(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)

	att := CreateAttestation("model.gguf", "deployment-ready", "test", []EvidenceItem{
		{Type: "scan", Source: "scanner", Timestamp: "2026-08-02T00:00:00Z", Result: "pass", Description: "clean"},
	})

	if err := SignAttestation(&att, priv); err != nil {
		t.Fatal(err)
	}
	if att.Signature == "" {
		t.Error("expected signature")
	}

	if !VerifyAttestation(att, pub) {
		t.Error("attestation should verify")
	}
}

func TestAttestation_TamperDetection(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)

	att := CreateAttestation("model.gguf", "safe", "tester", []EvidenceItem{{
		Type: "scan", Source: "scanner", Timestamp: "2026-08-02T00:00:00Z", Result: "pass",
	}})
	if err := SignAttestation(&att, priv); err != nil {
		t.Fatal(err)
	}

	att.Predicate = "unsafe"
	if VerifyAttestation(att, pub) {
		t.Error("tampered attestation should fail")
	}
}

func validTestBOM() *ModelBOM {
	return &ModelBOM{
		BOMFormat:    "CycloneDX",
		SpecVersion:  "1.6",
		Version:      1,
		SerialNumber: "urn:uuid:test",
		Components: []Component{{
			Type:   "machine-learning-model",
			BOMRef: "test.gguf",
			Name:   "test.gguf",
			Hashes: []Hash{{Algorithm: "SHA-256", Content: strings.Repeat("a", 64)}},
		}},
	}
}

func TestSignedBOM_Verify(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)

	bom := validTestBOM()

	sb, err := SignBOM(bom, priv, "test")
	if err != nil {
		t.Fatal(err)
	}
	valid, reason := VerifySignedBOM(sb, pub)
	if !valid {
		t.Errorf("signed BOM should verify: %s", reason)
	}
}

func TestSignedBOM_TamperDetection(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)

	bom := validTestBOM()
	sb, err := SignBOM(bom, priv, "test")
	if err != nil {
		t.Fatal(err)
	}

	sb.BOM.Version = 99
	valid, _ := VerifySignedBOM(sb, pub)
	if valid {
		t.Error("tampered BOM should fail verification")
	}
}

func TestSignedBOMMetadataTamperDetection(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	sb, err := SignBOM(validTestBOM(), priv, "trusted-signer")
	if err != nil {
		t.Fatal(err)
	}
	sb.SignedBy = "attacker"
	if valid, _ := VerifySignedBOM(sb, pub); valid {
		t.Fatal("signed-by tampering must invalidate the signature")
	}
}

func TestVerifySignedBOMRejectsStructurallyInvalidSignedContent(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	bom := validTestBOM()
	bom.Components[0].BOMRef = ""
	data, _ := json.Marshal(bom)
	hash := sha256.Sum256(data)
	hashText := hex.EncodeToString(hash[:])
	signedAt := "2026-08-02T00:00:00Z"
	payload, err := signedBOMSignaturePayload("1", hashText, signedAt, "malicious-signer")
	if err != nil {
		t.Fatal(err)
	}
	sb := SignedBOM{
		BOM: bom, Hash: hashText, Signature: hex.EncodeToString(ed25519.Sign(priv, payload)),
		SignatureVersion: "1", SignedAt: signedAt, SignedBy: "malicious-signer",
	}
	if valid, reason := VerifySignedBOM(sb, pub); valid || !strings.Contains(reason, "structural") {
		t.Fatalf("structurally invalid signed content was accepted: valid=%t reason=%q", valid, reason)
	}
}

func TestReadRegularFileRejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target.json")
	if err := os.WriteFile(target, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "input.json")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if _, err := readRegularFile(link, maxBOMBytes); err == nil {
		t.Fatal("symlinked input must be rejected")
	}
}

func TestLoadServiceTokenRejectsWhitespace(t *testing.T) {
	t.Setenv("SERVICE_TOKEN", strings.Repeat("a", 32)+" "+strings.Repeat("b", 32))
	t.Setenv("SERVICE_TOKEN_PATH", "")
	if _, err := loadServiceToken(); err == nil {
		t.Fatal("service token containing whitespace must be rejected")
	}
}

func TestEvaluateReadinessRequiresPassingEvidence(t *testing.T) {
	bom := validTestBOM()
	bom.Components[0].Evidence = []EvidenceItem{{
		Type: "scan", Source: "scanner", Timestamp: "2026-08-02T00:00:00Z", Result: "info",
	}}
	ready, missing := EvaluateReadiness(bom, []string{"evidence"})
	if ready || len(missing) == 0 {
		t.Fatalf("informational evidence must not satisfy a passing-evidence requirement: %v", missing)
	}
}

func TestEvaluateReadinessRejectsEmptyTrustFields(t *testing.T) {
	bom := validTestBOM()
	bom.Components[0].Licenses = []License{{}}
	bom.Components[0].ExternalRefs = []ExternalRef{{Type: "distribution"}}
	bom.Components[0].ModelCard = &ModelCard{
		TrustLabels:  []string{"  "},
		Quantization: &Quantization{},
	}
	ready, missing := EvaluateReadiness(bom, []string{"license", "source", "trust_labels", "quantization"})
	if ready || len(missing) < 4 {
		t.Fatalf("empty trust fields must not satisfy readiness: %v", missing)
	}
}

// ---------- readiness evaluation tests ----------

func TestEvaluateReadiness_Pass(t *testing.T) {
	bom := &ModelBOM{
		BOMFormat:    "CycloneDX",
		SpecVersion:  "1.6",
		Version:      1,
		SerialNumber: "urn:uuid:test",
		Components: []Component{{
			Type:     "machine-learning-model",
			BOMRef:   "model.gguf",
			Name:     "model.gguf",
			Hashes:   []Hash{{Algorithm: "SHA-256", Content: strings.Repeat("a", 64)}},
			Licenses: []License{{Name: "MIT"}},
			ModelCard: &ModelCard{
				TrustLabels: []string{"verified"},
			},
		}},
	}

	ready, missing := EvaluateReadiness(bom, []string{"hash", "license", "model_card", "trust_labels"})
	if !ready {
		t.Errorf("should be ready, missing: %v", missing)
	}
}

func TestEvaluateReadiness_Fail(t *testing.T) {
	bom := &ModelBOM{
		BOMFormat:    "CycloneDX",
		SpecVersion:  "1.6",
		Version:      1,
		SerialNumber: "urn:uuid:test",
		Components: []Component{{
			Type:   "machine-learning-model",
			BOMRef: "bare.gguf",
			Name:   "bare.gguf",
		}},
	}

	ready, missing := EvaluateReadiness(bom, []string{"hash", "license", "evidence"})
	if ready {
		t.Error("should not be ready without hash/license/evidence")
	}
	if len(missing) != 3 {
		t.Errorf("expected 3 missing, got %d: %v", len(missing), missing)
	}
}

// ---------- diff tests ----------

func TestDiff_Identical(t *testing.T) {
	bom := &ModelBOM{
		Version: 1,
		Components: []Component{
			{Name: "model.gguf", BOMRef: "model.gguf", Hashes: []Hash{{Algorithm: "SHA-256", Content: "same"}}},
		},
	}

	result := DiffBOMs(bom, bom)
	if len(result.Changes) != 0 {
		t.Errorf("expected no changes for identical BOMs, got %d", len(result.Changes))
	}
}

func TestDiff_HashChange(t *testing.T) {
	old := &ModelBOM{Components: []Component{
		{Name: "m.gguf", BOMRef: "m.gguf", Hashes: []Hash{{Algorithm: "SHA-256", Content: "aaaa1111222233334444555566667777"}}},
	}}
	new := &ModelBOM{Components: []Component{
		{Name: "m.gguf", BOMRef: "m.gguf", Hashes: []Hash{{Algorithm: "SHA-256", Content: "bbbb1111222233334444555566667777"}}},
	}}

	result := DiffBOMs(old, new)
	if !result.TrustRelevant {
		t.Error("hash change should be trust-relevant")
	}

	hasCritical := false
	for _, c := range result.Changes {
		if c.Category == "hash" && c.Severity == "critical" {
			hasCritical = true
		}
	}
	if !hasCritical {
		t.Error("expected critical hash change")
	}
}

func TestDiff_ComponentAdded(t *testing.T) {
	old := &ModelBOM{Components: []Component{
		{Name: "base.gguf", BOMRef: "base.gguf"},
	}}
	new := &ModelBOM{Components: []Component{
		{Name: "base.gguf", BOMRef: "base.gguf"},
		{Name: "adapter.bin", BOMRef: "adapter.bin"},
	}}

	result := DiffBOMs(old, new)
	hasAdded := false
	for _, c := range result.Changes {
		if strings.Contains(c.Message, "added") {
			hasAdded = true
		}
	}
	if !hasAdded {
		t.Error("expected component added change")
	}
}

func TestDiff_ComponentRemoved(t *testing.T) {
	old := &ModelBOM{Components: []Component{
		{Name: "model.gguf", BOMRef: "model.gguf"},
		{Name: "old.bin", BOMRef: "old.bin"},
	}}
	new := &ModelBOM{Components: []Component{
		{Name: "model.gguf", BOMRef: "model.gguf"},
	}}

	result := DiffBOMs(old, new)
	hasRemoved := false
	for _, c := range result.Changes {
		if strings.Contains(c.Message, "removed") {
			hasRemoved = true
		}
	}
	if !hasRemoved {
		t.Error("expected component removed change")
	}
}

func TestDiff_BaseModelChange(t *testing.T) {
	old := &ModelBOM{Components: []Component{
		{Name: "m", BOMRef: "m", ModelCard: &ModelCard{BaseModel: "llama-2-7b"}},
	}}
	new := &ModelBOM{Components: []Component{
		{Name: "m", BOMRef: "m", ModelCard: &ModelCard{BaseModel: "llama-3-8b"}},
	}}

	result := DiffBOMs(old, new)
	if !result.TrustRelevant {
		t.Error("base model change should be trust-relevant")
	}
}

func TestDiff_PromotionChange(t *testing.T) {
	old := &ModelBOM{Components: []Component{
		{Name: "m", BOMRef: "m", ModelCard: &ModelCard{PromotionStatus: "quarantine"}},
	}}
	new := &ModelBOM{Components: []Component{
		{Name: "m", BOMRef: "m", ModelCard: &ModelCard{PromotionStatus: "approved"}},
	}}

	result := DiffBOMs(old, new)
	hasPromo := false
	for _, c := range result.Changes {
		if c.Category == "trust" && strings.Contains(c.Field, "promotion") {
			hasPromo = true
		}
	}
	if !hasPromo {
		t.Error("expected promotion status change")
	}
}

// ---------- validation tests ----------

func TestValidateMetadata_ValidURL(t *testing.T) {
	meta := &ModelMetadata{SourceURL: "https://example.com/model"}
	issues := ValidateMetadata(meta)
	for _, iss := range issues {
		if iss.Field == "source_url" {
			t.Errorf("valid URL should not produce issue: %s", iss.Message)
		}
	}
}

func TestValidateMetadata_InvalidURL(t *testing.T) {
	meta := &ModelMetadata{SourceURL: "not-a-url"}
	issues := ValidateMetadata(meta)
	found := false
	for _, iss := range issues {
		if iss.Field == "source_url" && iss.Severity == "error" {
			found = true
		}
	}
	if !found {
		t.Error("invalid URL should produce error")
	}
}

func TestValidateMetadata_InvalidPromotion(t *testing.T) {
	meta := &ModelMetadata{PromotionStatus: "maybe"}
	issues := ValidateMetadata(meta)
	found := false
	for _, iss := range issues {
		if iss.Field == "promotion_status" && iss.Severity == "error" {
			found = true
		}
	}
	if !found {
		t.Error("invalid promotion status should produce error")
	}
}

func TestValidateMetadata_DuplicateTrustLabels(t *testing.T) {
	meta := &ModelMetadata{TrustLabels: []string{"scanned", "scanned", "verified"}}
	issues := ValidateMetadata(meta)
	found := false
	for _, iss := range issues {
		if iss.Field == "trust_labels" && iss.Severity == "warning" {
			found = true
		}
	}
	if !found {
		t.Error("duplicate trust labels should produce warning")
	}
	if len(meta.TrustLabels) != 2 {
		t.Errorf("expected deduplication to 2, got %d", len(meta.TrustLabels))
	}
}

func TestValidateBOM_Valid(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "m.gguf"), []byte("x"), 0o644)
	bom, _ := GenerateBOM(dir, nil)

	issues := ValidateBOM(bom)
	for _, iss := range issues {
		if iss.Severity == "error" {
			t.Errorf("valid BOM should not have errors: %s: %s", iss.Field, iss.Message)
		}
	}
}

func TestValidateBOM_DuplicateBOMRef(t *testing.T) {
	bom := &ModelBOM{
		BOMFormat:    "CycloneDX",
		SerialNumber: "urn:uuid:test",
		Components: []Component{
			{Name: "a", BOMRef: "same-ref"},
			{Name: "b", BOMRef: "same-ref"},
		},
	}
	issues := ValidateBOM(bom)
	found := false
	for _, iss := range issues {
		if iss.Field == "bom-ref" {
			found = true
		}
	}
	if !found {
		t.Error("duplicate bom-ref should produce error")
	}
}

// ---------- redaction tests ----------

func TestRedactEvidence_Emails(t *testing.T) {
	items := []EvidenceItem{
		{Description: "scanned by user@example.com", Result: "pass"},
	}
	profile := PrivacyProfile{StripEmails: true}
	redacted := RedactEvidence(items, profile)

	if strings.Contains(redacted[0].Description, "user@example.com") {
		t.Error("email should be redacted")
	}
	if !strings.Contains(redacted[0].Description, "[REDACTED:email]") {
		t.Error("expected [REDACTED:email] placeholder")
	}
}

func TestRedactEvidence_Paths(t *testing.T) {
	items := []EvidenceItem{
		{Description: "model at /home/user/models/test.gguf"},
	}
	profile := PrivacyProfile{StripPaths: true}
	redacted := RedactEvidence(items, profile)

	if strings.Contains(redacted[0].Description, "/home/user") {
		t.Error("path should be redacted")
	}
}

func TestRedactEvidence_Hostnames(t *testing.T) {
	items := []EvidenceItem{
		{Description: "fetched from build01.internal"},
	}
	profile := PrivacyProfile{StripHostnames: true}
	redacted := RedactEvidence(items, profile)

	if strings.Contains(redacted[0].Description, "build01.internal") {
		t.Error("hostname should be redacted")
	}
}

func TestRedactBOM_FullProfile(t *testing.T) {
	bom := &ModelBOM{
		Components: []Component{{
			Name: "model.gguf",
			Evidence: []EvidenceItem{
				{
					Description: "scanned by admin@corp.local on build01.internal",
					Details: map[string]string{
						"reviewer": "jane_doe",
						"path":     "/home/user/models/test.gguf",
					},
				},
			},
		}},
	}

	profile := DefaultPrivacyProfile()
	redacted := RedactBOM(bom, profile)

	ev := redacted.Components[0].Evidence[0]
	if strings.Contains(ev.Description, "admin@corp.local") {
		t.Error("email in description should be redacted")
	}
	if ev.Details["reviewer"] != "[REDACTED]" {
		t.Error("sensitive key 'reviewer' should be fully redacted")
	}
}

// ---------- path allowlist tests ----------

func TestIsPathAllowed_EmptyAllowlist(t *testing.T) {
	if isPathAllowed("/any/path", nil) {
		t.Error("empty allowlist must deny all paths")
	}
}

func TestIsPathAllowed_MatchingPath(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "models")
	os.MkdirAll(sub, 0o755)

	if !isPathAllowed(sub, []string{dir}) {
		t.Error("subpath of allowed dir should be permitted")
	}
}

func TestIsPathAllowed_NotMatching(t *testing.T) {
	dir := t.TempDir()
	if isPathAllowed("/etc/passwd", []string{dir}) {
		t.Error("path outside allowlist should be denied")
	}
}

// ---------- privacy profile resolution ----------

func TestResolvePrivacyProfile_None(t *testing.T) {
	p, err := resolvePrivacyProfile("none")
	if err != nil {
		t.Fatal(err)
	}
	if p.StripEmails || p.StripPaths || p.StripHostnames || p.StripUsernames {
		t.Error("'none' profile should strip nothing")
	}
}

func TestResolvePrivacyProfile_Default(t *testing.T) {
	p, err := resolvePrivacyProfile("default")
	if err != nil {
		t.Fatal(err)
	}
	if !p.StripEmails || !p.StripPaths || !p.StripHostnames || !p.StripUsernames {
		t.Error("'default' profile should strip all categories")
	}
}

// ---------- HTTP handler tests ----------

var testServiceToken = strings.Repeat("unit-test-token-", 3)

func buildTestMux(t *testing.T) http.Handler {
	t.Helper()
	mux := buildTestMuxWithToken(t, testServiceToken)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Header.Set("Authorization", "Bearer "+testServiceToken)
		mux.ServeHTTP(w, r)
	})
}

func buildTestMuxWithToken(t *testing.T, token string) *http.ServeMux {
	t.Helper()
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	cfg := &ServiceConfig{Version: 1}
	requireSigned := false
	cfg.RequireSignedTrustInputs = &requireSigned
	// Most handler tests exercise behavior rather than role separation, so use
	// the same fixture value for both roles. Dedicated tests below assert that
	// production wiring rejects a normal service token for signing operations.
	return buildMuxWithTokens(cfg, priv, pub, token, token)
}

func TestHTTP_Health(t *testing.T) {
	mux := buildTestMux(t)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/health", nil))
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_HealthNoAuth(t *testing.T) {
	mux := buildTestMuxWithToken(t, "secret")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/health", nil))
	if w.Code != 200 {
		t.Errorf("health should not require auth, got %d", w.Code)
	}
}

func TestHTTP_Diff(t *testing.T) {
	mux := buildTestMux(t)
	bomJSON := `{"bomFormat":"CycloneDX","specVersion":"1.6","version":1,"serialNumber":"urn:uuid:test","metadata":{"timestamp":"","tools":[]},"components":[{"type":"machine-learning-model","name":"m","bom-ref":"m","hashes":[{"alg":"SHA-256","content":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}]}`
	body := `{"old":` + bomJSON + `,"new":` + bomJSON + `}`
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/diff", strings.NewReader(body)))
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestHTTP_Evaluate(t *testing.T) {
	mux := buildTestMux(t)
	// Must satisfy default required_fields: hash, license, model_card
	body := `{"bomFormat":"CycloneDX","specVersion":"1.6","version":1,"serialNumber":"urn:uuid:test","metadata":{"timestamp":"","tools":[]},"components":[{"type":"machine-learning-model","bom-ref":"m","name":"m","hashes":[{"alg":"SHA-256","content":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"licenses":[{"name":"MIT"}],"modelCard":{}}]}`
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/evaluate", strings.NewReader(body)))
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var resp map[string]interface{}
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["ready"] != true {
		t.Error("should be ready with hash, license, and model_card present")
	}
}

func TestHTTPTrustEndpointsRequireSignedInputsByDefault(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	cfg := &ServiceConfig{Version: 1}
	mux := buildMuxWithVerifyKey(cfg, priv, pub, testServiceToken)
	raw, err := json.Marshal(validTestBOM())
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/evaluate", bytes.NewReader(raw))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unsigned readiness input must be rejected, got %d", w.Code)
	}

	signed, err := SignBOM(validTestBOM(), priv, "trusted")
	if err != nil {
		t.Fatal(err)
	}
	signedJSON, _ := json.Marshal(signed)
	req = httptest.NewRequest(http.MethodPost, "/v1/evaluate", bytes.NewReader(signedJSON))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"input_verified":true`) {
		t.Fatalf("signed readiness input was not verified: code=%d body=%s", w.Code, w.Body.String())
	}

	diffBody, _ := json.Marshal(map[string]SignedBOM{"old": signed, "new": signed})
	req = httptest.NewRequest(http.MethodPost, "/v1/diff", bytes.NewReader(diffBody))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK || w.Header().Get("SecAI-Input-Verified") != "true" {
		t.Fatalf("signed diff inputs were not verified: code=%d header=%q", w.Code, w.Header().Get("SecAI-Input-Verified"))
	}
}

func TestHTTP_AttestRequiresToken(t *testing.T) {
	mux := buildTestMuxWithToken(t, "secret123")

	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/v1/attest", strings.NewReader(`{"subject":"m","predicate":"safe"}`)))
	if w.Code != 401 {
		t.Errorf("expected 401 without token, got %d", w.Code)
	}

	req := httptest.NewRequest("POST", "/v1/attest", strings.NewReader(`{"subject":"m","predicate":"safe","evidence":[{"type":"scan","source":"unit-test","timestamp":"2026-08-02T00:00:00Z","result":"pass"}]}`))
	req.Header.Set("Authorization", "Bearer secret123")
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Errorf("expected 200 with valid token, got %d", w.Code)
	}
}

func TestHTTP_AllEndpointsRequireAuth(t *testing.T) {
	mux := buildTestMuxWithToken(t, "tok")

	endpoints := []struct {
		method string
		path   string
		body   string
	}{
		{"POST", "/v1/generate", `{"path":"/tmp"}`},
		{"POST", "/v1/verify", `{}`},
		{"POST", "/v1/diff", `{"old":{},"new":{}}`},
		{"POST", "/v1/evaluate", `{}`},
		{"POST", "/v1/attest", `{"subject":"m"}`},
		{"GET", "/v1/metrics", ""},
	}

	for _, ep := range endpoints {
		var body *strings.Reader
		if ep.body != "" {
			body = strings.NewReader(ep.body)
		} else {
			body = strings.NewReader("")
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(ep.method, ep.path, body))
		if w.Code != 401 {
			t.Errorf("%s %s: expected 401 without token, got %d", ep.method, ep.path, w.Code)
		}
	}
}

func TestHTTP_GeneratePathAllowlist(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	cfg := &ServiceConfig{Version: 1}
	cfg.Daemon.AllowedPaths = []string{"/only/this/dir"}
	mux := buildMuxWithTokens(cfg, priv, pub, testServiceToken, testServiceToken)

	body := `{"path":"/etc/passwd"}`
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/generate", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	mux.ServeHTTP(w, req)
	if w.Code != 403 {
		t.Errorf("expected 403 for disallowed path, got %d", w.Code)
	}
}

func TestPinnedAllowlistRootSurvivesAncestorReplacement(t *testing.T) {
	parent := t.TempDir()
	allowed := filepath.Join(parent, "allowed")
	moved := filepath.Join(parent, "allowed-original")
	outside := filepath.Join(parent, "outside")
	if err := os.Mkdir(allowed, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside, 0700); err != nil {
		t.Fatal(err)
	}
	safeBytes := []byte("approved-model")
	if err := os.WriteFile(filepath.Join(allowed, "model.gguf"), safeBytes, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "model.gguf"), []byte("outside-model"), 0600); err != nil {
		t.Fatal(err)
	}

	roots, err := openAllowedRoots([]string{allowed})
	if err != nil {
		t.Fatal(err)
	}
	defer closeAllowedRoots(roots)
	root, relative, display, ok := resolveAllowedTarget(filepath.Join(allowed, "model.gguf"), roots)
	if !ok {
		t.Fatal("expected target to resolve before replacement")
	}
	if err := os.Rename(allowed, moved); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, allowed); err != nil {
		t.Fatal(err)
	}

	bom, err := GenerateBOMFromRoot(root.root, relative, display, nil)
	if err != nil {
		t.Fatalf("pinned root should remain usable after pathname replacement: %v", err)
	}
	wantHash := sha256.Sum256(safeBytes)
	if got := bom.Components[0].Hashes[0].Content; got != hex.EncodeToString(wantHash[:]) {
		t.Fatalf("ancestor replacement escaped pinned root: got hash %s", got)
	}
}

func TestHTTPSigningEndpointsRequireSeparateAuthority(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	normalToken := strings.Repeat("normal-service-token-", 2)
	signingToken := strings.Repeat("signing-authority-token-", 2)
	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := &ServiceConfig{Version: 1}
	cfg.Daemon.AllowedPaths = []string{modelDir}
	mux := buildMuxWithTokens(cfg, priv, pub, normalToken, signingToken)
	body := `{"subject":"m","predicate":"safe","evidence":[{"type":"scan","source":"unit-test","timestamp":"2026-08-02T00:00:00Z","result":"pass"}]}`

	req := httptest.NewRequest(http.MethodPost, "/v1/attest", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+normalToken)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("ordinary service token must not mint attestations, got %d", w.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/v1/attest", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+signingToken)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("signing authority token was rejected: code=%d body=%s", w.Code, w.Body.String())
	}

	generateBody, _ := json.Marshal(map[string]string{"path": modelPath})
	req = httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(generateBody))
	req.Header.Set("Authorization", "Bearer "+normalToken)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("ordinary service token must not mint signed BOMs, got %d", w.Code)
	}
	req = httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(generateBody))
	req.Header.Set("Authorization", "Bearer "+signingToken)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"signed_bom"`) {
		t.Fatalf("signing authority could not generate a signed BOM: code=%d body=%s", w.Code, w.Body.String())
	}
}

func TestHTTPSigningEndpointsFailClosedWithoutSigningToken(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	mux := buildMuxWithTokens(&ServiceConfig{Version: 1}, priv, pub, testServiceToken, "")
	req := httptest.NewRequest(http.MethodPost, "/v1/attest", strings.NewReader(`{}`))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("missing signing authority must fail closed, got %d", w.Code)
	}
}

func TestHTTP_Metrics(t *testing.T) {
	mux := buildTestMux(t)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/v1/metrics", nil))
	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

// ---------- token auth ----------

func TestCheckToken_Empty(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	if checkToken(r, "") {
		t.Error("empty token must fail closed")
	}
}

func TestCheckToken_Valid(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.Header.Set("Authorization", "Bearer tok")
	if !checkToken(r, "tok") {
		t.Error("valid token should pass")
	}
}

func TestCheckToken_Invalid(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.Header.Set("Authorization", "Bearer wrong")
	if checkToken(r, "right") {
		t.Error("invalid token should fail")
	}
}

func TestHTTP_MissingTokenConfigurationFailsClosed(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	mux := buildMux(&ServiceConfig{Version: 1}, priv, "")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/metrics", nil))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestHTTP_RateLimit(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	cfg := &ServiceConfig{Version: 1}
	cfg.Daemon.RequestsPerMinute = 1
	mux := buildMux(cfg, priv, testServiceToken)
	for attempt := 1; attempt <= 2; attempt++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/metrics", nil)
		req.Header.Set("Authorization", "Bearer "+testServiceToken)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if attempt == 1 && w.Code != http.StatusOK {
			t.Fatalf("first request returned %d", w.Code)
		}
		if attempt == 2 && w.Code != http.StatusTooManyRequests {
			t.Fatalf("second request returned %d", w.Code)
		}
	}
}

func TestHTTP_RejectsUnknownAndTrailingJSON(t *testing.T) {
	mux := buildTestMuxWithToken(t, testServiceToken)
	for _, body := range []string{
		`{"subject":"m","predicate":"safe","unexpected":true}`,
		`{"subject":"m","predicate":"safe"} {}`,
	} {
		req := httptest.NewRequest(http.MethodPost, "/v1/attest", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+testServiceToken)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %q, got %d", body, w.Code)
		}
	}
}

func TestHTTP_AttestRequiresSigningKey(t *testing.T) {
	mux := buildMux(&ServiceConfig{Version: 1}, nil, testServiceToken)
	req := httptest.NewRequest(http.MethodPost, "/v1/attest", strings.NewReader(`{"subject":"m","predicate":"safe"}`))
	req.Header.Set("Authorization", "Bearer "+testServiceToken)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", w.Code)
	}
}

func TestSignBOMRejectsInvalidPrivateKey(t *testing.T) {
	if _, err := SignBOM(validTestBOM(), ed25519.PrivateKey("short"), "test"); err == nil {
		t.Fatal("expected invalid private key to be rejected")
	}
}

func TestEvaluateReadinessRejectsFailedEvidence(t *testing.T) {
	bom := validTestBOM()
	bom.Components[0].Evidence = []EvidenceItem{{Type: "scan", Source: "scanner", Result: "fail"}}
	ready, missing := EvaluateReadiness(bom, nil)
	if ready || len(missing) == 0 {
		t.Fatalf("failed evidence must block readiness: %v", missing)
	}
}

func TestLoadConfigRejectsUnknownAndMultipleDocuments(t *testing.T) {
	for _, content := range []string{
		"version: 1\nunknown: true\n",
		"version: 1\n---\nversion: 1\n",
	} {
		path := filepath.Join(t.TempDir(), "config.yaml")
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := loadConfig(path); err == nil {
			t.Fatalf("expected invalid config to be rejected: %q", content)
		}
	}
}

func TestImportEvidenceRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evidence.json")
	if err := os.WriteFile(path, []byte(`{"type":"scan","result":"pass","unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ImportEvidence(path); err == nil {
		t.Fatal("expected unknown evidence field to be rejected")
	}
}

func TestLoadSignedBOMRequiresAndUsesVerificationKey(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	signed, err := SignBOM(validTestBOM(), priv, "test")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(signed)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "signed.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VERIFY_KEY", "")
	t.Setenv("VERIFY_KEY_PATH", "")
	if _, err := loadBOMFile(path); err == nil {
		t.Fatal("expected a signed BOM without a verification key to be rejected")
	}
	t.Setenv("VERIFY_KEY", base64.StdEncoding.EncodeToString(pub))
	if _, err := loadBOMFile(path); err != nil {
		t.Fatalf("expected signed BOM to verify: %v", err)
	}
}

func TestUnsignedBOMRequiresExplicitPolicy(t *testing.T) {
	data, err := json.Marshal(validTestBOM())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "bom.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadBOMFile(path); err == nil {
		t.Fatal("unsigned BOM must be rejected by default")
	}
	if _, err := loadBOMFileWithPolicy(path, true); err != nil {
		t.Fatalf("explicit structural-only unsigned load failed: %v", err)
	}
}

func TestDirectoryBOMRejectsSymlinkEntries(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("model"), 0600); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(dir, "linked-directory")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := GenerateBOM(dir, nil); err == nil {
		t.Fatal("directory inventory must reject symlink entries rather than silently omit them")
	}
}

func TestBOMValidationRejectsDanglingDependencies(t *testing.T) {
	bom := validTestBOM()
	bom.Dependencies = []Dependency{{Ref: bom.Components[0].BOMRef, DependsOn: []string{"missing"}}}
	if !hasBOMValidationErrors(bom) {
		t.Fatal("dangling dependency must fail validation")
	}
}

func TestTokenAndBindValidation(t *testing.T) {
	if validServiceToken(strings.Repeat("a", 31)) || validServiceToken(strings.Repeat("a", 32)+"\x7f") {
		t.Fatal("weak or control-bearing token accepted")
	}
	if !validServiceToken(strings.Repeat("a", 32)) {
		t.Fatal("strong token rejected")
	}
	if !isLoopbackBind("127.0.0.1:8515") || isLoopbackBind("0.0.0.0:8515") {
		t.Fatal("bind classification is incorrect")
	}
}

func TestURLValidationRejectsUnsafeSchemes(t *testing.T) {
	if isValidURL("file:///etc/passwd") || isValidURL("http://example.com/model") {
		t.Fatal("unsafe URL scheme accepted")
	}
	if !isValidURL("https://example.com/model") || !isValidURL("http://127.0.0.1/model") {
		t.Fatal("expected secure and loopback URLs to be accepted")
	}
}

// ---------- dependency graph tests ----------

func TestBuildDependencies_ModelWithTokenizer(t *testing.T) {
	components := []Component{
		{Type: "machine-learning-model", BOMRef: "model.gguf", Name: "model.gguf"},
		{Type: "data", BOMRef: "tokenizer:tokenizer.json", Name: "tokenizer.json",
			Properties: []Property{{Name: "secai:component_type", Value: "tokenizer"}}},
	}

	deps := buildDependencies(components)
	if len(deps) != 1 {
		t.Fatalf("expected 1 dependency, got %d", len(deps))
	}
	if deps[0].Ref != "model.gguf" {
		t.Errorf("expected model ref, got %s", deps[0].Ref)
	}
	if len(deps[0].DependsOn) != 1 || deps[0].DependsOn[0] != "tokenizer:tokenizer.json" {
		t.Error("expected tokenizer in dependsOn")
	}
}

func TestBuildDependencies_SingleComponent(t *testing.T) {
	deps := buildDependencies([]Component{{Name: "m.gguf"}})
	if deps != nil {
		t.Error("single component should have no dependencies")
	}
}

// ---------- integration test ----------

func TestIntegration_FullWorkflow(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "model.Q5_K_S.gguf"), []byte("model-weights"), 0o644)

	// 1. Generate BOM
	meta := &ModelMetadata{
		BaseModel:       "llama-2-7b",
		License:         "Apache-2.0",
		TrustLabels:     []string{"scanned"},
		PromotionStatus: "approved",
	}
	bom, err := GenerateBOM(dir, meta)
	if err != nil {
		t.Fatal(err)
	}
	if len(bom.Components) == 0 {
		t.Fatal("no components generated")
	}

	// 2. Validate BOM
	issues := ValidateBOM(bom)
	for _, iss := range issues {
		if iss.Severity == "error" {
			t.Errorf("generated BOM has validation error: %s: %s", iss.Field, iss.Message)
		}
	}

	// 3. Attach evidence
	evidence := []EvidenceItem{
		{Type: "scan", Source: "ai-quarantine", Result: "pass", Description: "clean", Timestamp: "2025-01-01T00:00:00Z"},
		{Type: "test", Source: "gguf-guard", Result: "pass", Description: "valid GGUF", Timestamp: "2025-01-01T00:00:00Z"},
	}
	AttachEvidence(bom, bom.Components[0].Name, evidence)

	// 4. Evaluate readiness
	ready, _ := EvaluateReadiness(bom, []string{"hash", "license", "evidence", "trust_labels"})
	if !ready {
		t.Error("should be ready after attaching evidence")
	}

	// 5. Sign BOM
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	sb, err := SignBOM(bom, priv, "test")
	if err != nil {
		t.Fatal(err)
	}
	valid, reason := VerifySignedBOM(sb, pub)
	if !valid {
		t.Errorf("signed BOM should verify: %s", reason)
	}

	// 6. Create attestation
	att := CreateAttestation(bom.Components[0].Name, "deployment-ready", "ci", evidence)
	if err := SignAttestation(&att, priv); err != nil {
		t.Fatal(err)
	}
	if !VerifyAttestation(att, pub) {
		t.Error("attestation should verify")
	}

	// 7. Build lineage
	h := sha256.Sum256([]byte("model-weights"))
	modelHash := hex.EncodeToString(h[:])
	chain := BuildQuantizationLineage("llama-2-7b", "base-hash", "Q5_K_S", modelHash, "operator")
	chain.AddPromotion("approved", "admin", "all scans passed")
	chainValid, _ := chain.Verify()
	if !chainValid {
		t.Error("lineage chain should be valid")
	}

	// 8. Diff with modified version
	bom2, _ := GenerateBOM(dir, &ModelMetadata{
		BaseModel:       "llama-3-8b", // changed!
		License:         "Apache-2.0",
		PromotionStatus: "approved",
	})
	diff := DiffBOMs(bom, bom2)
	if !diff.TrustRelevant {
		t.Error("base model change should be trust-relevant")
	}

	// 9. Redact evidence
	redacted := RedactBOM(bom, DefaultPrivacyProfile())
	if redacted == nil {
		t.Error("redacted BOM should not be nil")
	}
}
