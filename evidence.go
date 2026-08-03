package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// EvidenceItem records a scan result, test outcome, or attestation.
type EvidenceItem struct {
	Type        string            `json:"type" yaml:"type"`     // scan, test, attestation, review
	Source      string            `json:"source" yaml:"source"` // ai-quarantine, gguf-guard, manual, etc.
	Timestamp   string            `json:"timestamp" yaml:"timestamp"`
	Result      string            `json:"result" yaml:"result"` // pass, fail, warning, info
	Description string            `json:"description" yaml:"description"`
	Details     map[string]string `json:"details,omitempty" yaml:"details,omitempty"`
	Hash        string            `json:"hash,omitempty" yaml:"hash,omitempty"`
}

// Attestation is a signed statement about a model's trustworthiness.
type Attestation struct {
	Subject   string         `json:"subject"`   // model name or hash
	Predicate string         `json:"predicate"` // what is being attested
	Issuer    string         `json:"issuer"`    // who is attesting
	IssuedAt  string         `json:"issued_at"`
	Evidence  []EvidenceItem `json:"evidence"`
	Hash      string         `json:"hash"`
	Signature string         `json:"signature,omitempty"`
}

// SignedBOM wraps a BOM with a signature.
type SignedBOM struct {
	BOM              *ModelBOM `json:"bom"`
	Hash             string    `json:"hash"`
	Signature        string    `json:"signature"`
	SignatureVersion string    `json:"signature_version"`
	SignedAt         string    `json:"signed_at"`
	SignedBy         string    `json:"signed_by"`
}

// ImportEvidence reads evidence items from a JSON file (e.g. quarantine scan results).
func ImportEvidence(path string) ([]EvidenceItem, error) {
	data, err := readRegularFile(path, maxEvidenceBytes)
	if err != nil {
		return nil, fmt.Errorf("read evidence file: %w", err)
	}

	var items []EvidenceItem
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("[")) {
		if err := decodeStrictJSON(data, &items); err != nil {
			return nil, fmt.Errorf("parse evidence: %w", err)
		}
	} else {
		var item EvidenceItem
		if err := decodeStrictJSON(data, &item); err != nil {
			return nil, fmt.Errorf("parse evidence: %w", err)
		}
		items = []EvidenceItem{item}
	}
	if len(items) == 0 || len(items) > 10000 {
		return nil, errors.New("evidence must contain between 1 and 10000 items")
	}
	for _, item := range items {
		for _, issue := range validateEvidenceItem(item) {
			if issue.Severity == "error" {
				return nil, fmt.Errorf("invalid evidence: %s", issue.Message)
			}
		}
	}
	return items, nil
}

// AttachEvidence adds evidence items to a BOM component.
func AttachEvidence(bom *ModelBOM, componentRef string, evidence []EvidenceItem) bool {
	for i := range bom.Components {
		if bom.Components[i].BOMRef == componentRef || bom.Components[i].Name == componentRef {
			bom.Components[i].Evidence = append(bom.Components[i].Evidence, evidence...)
			return true
		}
	}
	return false
}

// CreateAttestation builds an attestation for a model.
func CreateAttestation(subject, predicate, issuer string, evidence []EvidenceItem) Attestation {
	att := Attestation{
		Subject:   subject,
		Predicate: predicate,
		Issuer:    issuer,
		IssuedAt:  time.Now().UTC().Format(time.RFC3339),
		Evidence:  evidence,
	}

	att.Hash = computeAttestationHash(att)
	return att
}

// SignAttestation signs an attestation with an Ed25519 key.
func SignAttestation(att *Attestation, privKey ed25519.PrivateKey) error {
	if att == nil {
		return errors.New("attestation is nil")
	}
	if len(privKey) != ed25519.PrivateKeySize {
		return errors.New("invalid Ed25519 private key")
	}
	if strings.TrimSpace(att.Subject) == "" || strings.TrimSpace(att.Predicate) == "" || strings.TrimSpace(att.Issuer) == "" {
		return errors.New("attestation subject, predicate, and issuer are required")
	}
	if len(att.Subject) > 1024 || len(att.Predicate) > 1024 || len(att.Issuer) > 1024 ||
		!isValidRFC3339(att.IssuedAt) || len(att.Evidence) == 0 || len(att.Evidence) > 10000 {
		return errors.New("attestation fields are outside the supported limits")
	}
	for _, evidence := range att.Evidence {
		for _, issue := range validateEvidenceItem(evidence) {
			if issue.Severity == "error" {
				return fmt.Errorf("invalid attestation evidence: %s", issue.Message)
			}
		}
	}
	expected := computeAttestationHash(*att)
	if att.Hash != expected {
		return errors.New("attestation hash does not match its content")
	}
	hashBytes, err := hex.DecodeString(att.Hash)
	if err != nil || len(hashBytes) != sha256.Size {
		return errors.New("invalid attestation hash")
	}
	payload, err := attestationSignaturePayload(att.Hash)
	if err != nil {
		return err
	}
	sig := ed25519.Sign(privKey, payload)
	att.Signature = hex.EncodeToString(sig)
	return nil
}

// VerifyAttestation checks an attestation's signature.
func VerifyAttestation(att Attestation, pubKey ed25519.PublicKey) bool {
	if att.Signature == "" || len(pubKey) != ed25519.PublicKeySize {
		return false
	}
	if strings.TrimSpace(att.Subject) == "" || strings.TrimSpace(att.Predicate) == "" ||
		strings.TrimSpace(att.Issuer) == "" || len(att.Subject) > 1024 ||
		len(att.Predicate) > 1024 || len(att.Issuer) > 1024 ||
		!isValidRFC3339(att.IssuedAt) || len(att.Evidence) == 0 || len(att.Evidence) > 10000 {
		return false
	}
	for _, evidence := range att.Evidence {
		for _, issue := range validateEvidenceItem(evidence) {
			if issue.Severity == "error" {
				return false
			}
		}
	}

	expected := computeAttestationHash(att)
	if att.Hash != expected {
		return false
	}

	hashBytes, err := hex.DecodeString(att.Hash)
	if err != nil {
		return false
	}

	sig, err := hex.DecodeString(att.Signature)
	if err != nil {
		return false
	}

	payload, err := attestationSignaturePayload(att.Hash)
	return err == nil && len(hashBytes) == sha256.Size && len(sig) == ed25519.SignatureSize && ed25519.Verify(pubKey, payload, sig)
}

// SignBOM creates a signed BOM wrapper.
func SignBOM(bom *ModelBOM, privKey ed25519.PrivateKey, signer string) (SignedBOM, error) {
	if bom == nil || hasBOMValidationErrors(bom) {
		return SignedBOM{}, errors.New("cannot sign an invalid BOM")
	}
	if len(privKey) != ed25519.PrivateKeySize {
		return SignedBOM{}, errors.New("invalid Ed25519 private key")
	}
	if strings.TrimSpace(signer) == "" || len(signer) > 1024 {
		return SignedBOM{}, errors.New("signer is required")
	}
	data, err := json.Marshal(bom)
	if err != nil {
		return SignedBOM{}, err
	}
	h := sha256.Sum256(data)
	hashStr := hex.EncodeToString(h[:])

	signedAt := time.Now().UTC().Format(time.RFC3339)
	const signatureVersion = "1"
	payload, err := signedBOMSignaturePayload(signatureVersion, hashStr, signedAt, signer)
	if err != nil {
		return SignedBOM{}, err
	}
	sig := ed25519.Sign(privKey, payload)

	return SignedBOM{
		BOM:              bom,
		Hash:             hashStr,
		Signature:        hex.EncodeToString(sig),
		SignatureVersion: signatureVersion,
		SignedAt:         signedAt,
		SignedBy:         signer,
	}, nil
}

// VerifySignedBOM checks a signed BOM's integrity and signature.
func VerifySignedBOM(sb SignedBOM, pubKey ed25519.PublicKey) (bool, string) {
	if sb.BOM == nil || sb.Signature == "" || len(pubKey) != ed25519.PublicKeySize {
		return false, "missing signature or public key"
	}
	if sb.SignatureVersion != "1" || !isValidRFC3339(sb.SignedAt) || strings.TrimSpace(sb.SignedBy) == "" || len(sb.SignedBy) > 1024 {
		return false, "invalid signed BOM metadata"
	}
	if hasBOMValidationErrors(sb.BOM) {
		return false, "BOM failed structural validation"
	}

	data, err := json.Marshal(sb.BOM)
	if err != nil {
		return false, "BOM cannot be encoded"
	}
	h := sha256.Sum256(data)
	hashStr := hex.EncodeToString(h[:])

	if hashStr != sb.Hash {
		return false, "BOM hash mismatch: content has been modified"
	}

	sig, err := hex.DecodeString(sb.Signature)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return false, "invalid signature encoding"
	}

	payload, err := signedBOMSignaturePayload(sb.SignatureVersion, sb.Hash, sb.SignedAt, sb.SignedBy)
	if err != nil {
		return false, "signed BOM metadata cannot be encoded"
	}
	if !ed25519.Verify(pubKey, payload, sig) {
		return false, "signature verification failed"
	}

	return true, "valid"
}

func signedBOMSignaturePayload(version, hash, signedAt, signedBy string) ([]byte, error) {
	return json.Marshal(struct {
		Domain   string `json:"domain"`
		Version  string `json:"version"`
		Hash     string `json:"hash"`
		SignedAt string `json:"signed_at"`
		SignedBy string `json:"signed_by"`
	}{
		Domain: "SecAI-Hub/ai-model-bom", Version: version, Hash: hash,
		SignedAt: signedAt, SignedBy: signedBy,
	})
}

func attestationSignaturePayload(hash string) ([]byte, error) {
	return json.Marshal(struct {
		Domain  string `json:"domain"`
		Version string `json:"version"`
		Hash    string `json:"hash"`
	}{Domain: "SecAI-Hub/ai-model-bom/attestation", Version: "1", Hash: hash})
}

// ---------- helpers ----------

func computeAttestationHash(att Attestation) string {
	data, _ := json.Marshal(map[string]interface{}{
		"subject":   att.Subject,
		"predicate": att.Predicate,
		"issuer":    att.Issuer,
		"issued_at": att.IssuedAt,
		"evidence":  att.Evidence,
	})
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

// EvaluateReadiness checks if a BOM meets minimum deployment criteria.
func EvaluateReadiness(bom *ModelBOM, requiredFields []string) (bool, []string) {
	var missing []string
	if bom == nil {
		return false, []string{"BOM is nil"}
	}
	for _, issue := range ValidateBOM(bom) {
		if issue.Severity == "error" {
			missing = append(missing, fmt.Sprintf("invalid %s: %s", issue.Field, issue.Message))
		}
	}

	if len(bom.Components) == 0 {
		missing = append(missing, "no components in BOM")
	}

	modelCount := 0
	for _, comp := range bom.Components {
		if comp.Type != "machine-learning-model" {
			continue
		}
		modelCount++
		for _, ev := range comp.Evidence {
			if strings.EqualFold(ev.Result, "fail") {
				missing = append(missing, fmt.Sprintf("%s: failed evidence from %s", comp.Name, ev.Source))
			}
		}
		if comp.ModelCard != nil && comp.ModelCard.PromotionStatus != "" &&
			!strings.EqualFold(comp.ModelCard.PromotionStatus, "approved") {
			missing = append(missing, fmt.Sprintf("%s: promotion status is %s", comp.Name, comp.ModelCard.PromotionStatus))
		}

		for _, field := range requiredFields {
			switch field {
			case "hash":
				if !hasValidSHA256(comp.Hashes) {
					missing = append(missing, fmt.Sprintf("%s: no hash", comp.Name))
				}
			case "license":
				if !hasValidLicense(comp.Licenses) {
					missing = append(missing, fmt.Sprintf("%s: no license", comp.Name))
				}
			case "model_card":
				if comp.ModelCard == nil {
					missing = append(missing, fmt.Sprintf("%s: no model card", comp.Name))
				}
			case "evidence":
				if !hasPassingEvidence(comp.Evidence) {
					missing = append(missing, fmt.Sprintf("%s: no valid passing evidence", comp.Name))
				}
			case "trust_labels":
				if comp.ModelCard == nil || !hasValidTrustLabel(comp.ModelCard.TrustLabels) {
					missing = append(missing, fmt.Sprintf("%s: no trust labels", comp.Name))
				}
			case "quantization":
				if comp.ModelCard == nil || comp.ModelCard.Quantization == nil || strings.TrimSpace(comp.ModelCard.Quantization.Method) == "" {
					missing = append(missing, fmt.Sprintf("%s: no quantization info", comp.Name))
				}
			case "source":
				if !hasValidExternalReference(comp.ExternalRefs) {
					missing = append(missing, fmt.Sprintf("%s: no source URL", comp.Name))
				}
			case "promotion_status":
				if comp.ModelCard == nil || !strings.EqualFold(comp.ModelCard.PromotionStatus, "approved") {
					missing = append(missing, fmt.Sprintf("%s: promotion status is not approved", comp.Name))
				}
			default:
				missing = append(missing, fmt.Sprintf("unsupported required field: %s", field))
			}
		}
	}
	if modelCount == 0 {
		missing = append(missing, "no machine-learning-model component in BOM")
	}

	return len(missing) == 0, deduplicateStrings(missing)
}

func hasValidLicense(licenses []License) bool {
	for _, license := range licenses {
		if strings.TrimSpace(license.ID) != "" || strings.TrimSpace(license.Name) != "" ||
			(license.URL != "" && isValidURL(license.URL)) {
			return true
		}
	}
	return false
}

func hasValidTrustLabel(labels []string) bool {
	for _, label := range labels {
		if strings.TrimSpace(label) != "" && len(label) <= 256 {
			return true
		}
	}
	return false
}

func hasValidExternalReference(references []ExternalRef) bool {
	for _, reference := range references {
		if strings.TrimSpace(reference.Type) != "" && isValidURL(reference.URL) {
			return true
		}
	}
	return false
}

func hasPassingEvidence(items []EvidenceItem) bool {
	for _, item := range items {
		if item.Result != "pass" {
			continue
		}
		valid := true
		for _, issue := range validateEvidenceItem(item) {
			if issue.Severity == "error" {
				valid = false
				break
			}
		}
		if valid {
			return true
		}
	}
	return false
}

func hasValidSHA256(hashes []Hash) bool {
	for _, hash := range hashes {
		if hash.Algorithm != "SHA-256" || len(hash.Content) != sha256.Size*2 {
			continue
		}
		decoded, err := hex.DecodeString(hash.Content)
		if err == nil && len(decoded) == sha256.Size {
			return true
		}
	}
	return false
}
