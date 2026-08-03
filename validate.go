package main

import (
	"encoding/hex"
	"fmt"
	"math"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"
)

// ---------- metadata validation ----------

// ValidationIssue describes a problem found during validation.
type ValidationIssue struct {
	Field    string `json:"field"`
	Severity string `json:"severity"` // error, warning
	Message  string `json:"message"`
}

// ValidateMetadata checks user-supplied metadata for correctness.
func ValidateMetadata(meta *ModelMetadata) []ValidationIssue {
	if meta == nil {
		return nil
	}

	var issues []ValidationIssue
	if len(meta.Name) > 4096 || len(meta.Version) > 4096 || len(meta.Description) > 64<<10 ||
		len(meta.BaseModel) > 4096 || len(meta.BaseModelHash) > 1024 || len(meta.ModelFamily) > 4096 ||
		len(meta.License) > 4096 || len(meta.SourceURL) > 8192 || len(meta.TrustLabels) > 1024 ||
		len(meta.PromotionReason) > 64<<10 || len(meta.QuantizedBy) > 4096 ||
		strings.ContainsAny(meta.Name+meta.Version+meta.BaseModel+meta.ModelFamily+meta.QuantizedBy, "\x00") {
		issues = append(issues, ValidationIssue{Field: "metadata", Severity: "error", Message: "metadata fields exceed supported limits"})
	}
	if meta.BaseModelHash != "" {
		decoded, err := hex.DecodeString(meta.BaseModelHash)
		if err != nil || len(decoded) != 32 {
			issues = append(issues, ValidationIssue{Field: "base_model_hash", Severity: "error", Message: "base model hash must be SHA-256"})
		}
	}

	if meta.SourceURL != "" {
		if !isValidURL(meta.SourceURL) {
			issues = append(issues, ValidationIssue{
				Field: "source_url", Severity: "error",
				Message: fmt.Sprintf("invalid URL: %q", meta.SourceURL),
			})
		}
	}

	if meta.License != "" {
		if !isKnownSPDX(meta.License) {
			issues = append(issues, ValidationIssue{
				Field: "license", Severity: "warning",
				Message: fmt.Sprintf("license %q is not a recognized SPDX identifier", meta.License),
			})
		}
	}

	if meta.PromotionStatus != "" {
		if !isValidPromotionStatus(meta.PromotionStatus) {
			issues = append(issues, ValidationIssue{
				Field: "promotion_status", Severity: "error",
				Message: fmt.Sprintf("invalid promotion status %q; expected one of: quarantine, pending, approved, rejected, revoked", meta.PromotionStatus),
			})
		}
	}

	if meta.QuantizedAt != "" {
		if !isValidRFC3339(meta.QuantizedAt) {
			issues = append(issues, ValidationIssue{
				Field: "quantized_at", Severity: "error",
				Message: fmt.Sprintf("invalid RFC3339 timestamp: %q", meta.QuantizedAt),
			})
		}
	}

	// Deduplicate trust labels
	if len(meta.TrustLabels) > 0 {
		seen := map[string]bool{}
		var deduped []string
		for _, l := range meta.TrustLabels {
			l = strings.TrimSpace(l)
			if l == "" {
				continue
			}
			if seen[l] {
				issues = append(issues, ValidationIssue{
					Field: "trust_labels", Severity: "warning",
					Message: fmt.Sprintf("duplicate trust label: %q", l),
				})
				continue
			}
			seen[l] = true
			deduped = append(deduped, l)
		}
		meta.TrustLabels = deduped
	}

	return issues
}

// ---------- BOM validation ----------

// ValidateBOM checks a generated BOM for structural issues.
func ValidateBOM(bom *ModelBOM) []ValidationIssue {
	if bom == nil {
		return []ValidationIssue{{Field: "bom", Severity: "error", Message: "BOM is nil"}}
	}

	var issues []ValidationIssue

	if bom.BOMFormat != "CycloneDX" {
		issues = append(issues, ValidationIssue{
			Field: "bomFormat", Severity: "error",
			Message: fmt.Sprintf("expected CycloneDX, got %q", bom.BOMFormat),
		})
	}
	if bom.SpecVersion != "1.6" {
		issues = append(issues, ValidationIssue{
			Field: "specVersion", Severity: "error",
			Message: fmt.Sprintf("expected 1.6, got %q", bom.SpecVersion),
		})
	}
	if bom.Version < 1 {
		issues = append(issues, ValidationIssue{Field: "version", Severity: "error", Message: "version must be positive"})
	}
	if len(bom.Components) == 0 || len(bom.Components) > 100000 {
		issues = append(issues, ValidationIssue{Field: "components", Severity: "error", Message: "component count is outside the supported range"})
	}

	if bom.SerialNumber == "" || len(bom.SerialNumber) > 1024 || strings.ContainsAny(bom.SerialNumber, "\r\n\x00") {
		issues = append(issues, ValidationIssue{
			Field: "serialNumber", Severity: "error",
			Message: "missing serial number",
		})
	}
	if bom.Metadata.Timestamp != "" && !isValidRFC3339(bom.Metadata.Timestamp) {
		issues = append(issues, ValidationIssue{Field: "metadata.timestamp", Severity: "error", Message: "metadata timestamp must be RFC3339"})
	}
	if len(bom.Metadata.Tools) > 128 {
		issues = append(issues, ValidationIssue{Field: "metadata.tools", Severity: "error", Message: "too many metadata tools"})
	}
	for _, tool := range bom.Metadata.Tools {
		if len(tool.Vendor) > 1024 || len(tool.Name) > 1024 || len(tool.Version) > 1024 || strings.ContainsAny(tool.Vendor+tool.Name+tool.Version, "\x00") {
			issues = append(issues, ValidationIssue{Field: "metadata.tools", Severity: "error", Message: "metadata tool fields exceed supported limits"})
		}
	}

	// Unique bom-ref validation
	refs := map[string]int{}
	for _, comp := range bom.Components {
		if comp.BOMRef != "" {
			refs[comp.BOMRef]++
		}
	}
	for ref, count := range refs {
		if count > 1 {
			issues = append(issues, ValidationIssue{
				Field: "bom-ref", Severity: "error",
				Message: fmt.Sprintf("duplicate bom-ref: %q (appears %d times)", ref, count),
			})
		}
	}

	// Component-level validation
	for _, comp := range bom.Components {
		if comp.Type == "" || len(comp.Type) > 256 || strings.ContainsAny(comp.Type, "\r\n\x00") {
			issues = append(issues, ValidationIssue{Field: "component.type", Severity: "error", Message: "component has empty type"})
		}
		if comp.BOMRef == "" || len(comp.BOMRef) > 4096 || strings.ContainsAny(comp.BOMRef, "\r\n\x00") {
			issues = append(issues, ValidationIssue{Field: "component.bom-ref", Severity: "error", Message: "component has empty bom-ref"})
		}
		if comp.Name == "" || len(comp.Name) > 4096 || strings.ContainsAny(comp.Name, "\r\n\x00") {
			issues = append(issues, ValidationIssue{
				Field: "component.name", Severity: "error",
				Message: "component has empty name",
			})
		}
		if len(comp.Version) > 4096 || len(comp.Description) > 64<<10 ||
			len(comp.Hashes) > 128 || len(comp.Licenses) > 128 || len(comp.ExternalRefs) > 1024 ||
			len(comp.Properties) > 10000 || len(comp.Evidence) > 10000 {
			issues = append(issues, ValidationIssue{Field: "component", Severity: "error", Message: "component fields exceed supported limits"})
		}
		for _, hash := range comp.Hashes {
			if hash.Algorithm != "SHA-256" {
				issues = append(issues, ValidationIssue{Field: "component.hash", Severity: "error", Message: fmt.Sprintf("unsupported hash algorithm %q", hash.Algorithm)})
				continue
			}
			decoded, err := hex.DecodeString(hash.Content)
			if err != nil || len(decoded) != 32 {
				issues = append(issues, ValidationIssue{Field: "component.hash", Severity: "error", Message: "SHA-256 hash must contain 64 hexadecimal characters"})
			}
		}

		for _, lic := range comp.Licenses {
			if len(lic.ID) > 1024 || len(lic.Name) > 4096 || len(lic.URL) > 8192 || strings.ContainsAny(lic.ID+lic.Name+lic.URL, "\x00") {
				issues = append(issues, ValidationIssue{Field: "component.license", Severity: "error", Message: "license fields exceed supported limits"})
				continue
			}
			if strings.TrimSpace(lic.ID) == "" && strings.TrimSpace(lic.Name) == "" && strings.TrimSpace(lic.URL) == "" {
				issues = append(issues, ValidationIssue{
					Field: "component.license", Severity: "error", Message: "license entry is empty",
				})
				continue
			}
			if lic.ID != "" && !isKnownSPDX(lic.ID) {
				issues = append(issues, ValidationIssue{
					Field: "component.license", Severity: "warning",
					Message: fmt.Sprintf("license ID %q is not a recognized SPDX identifier", lic.ID),
				})
			}
			if lic.URL != "" && !isValidURL(lic.URL) {
				issues = append(issues, ValidationIssue{
					Field: "component.license.url", Severity: "error", Message: "license URL is invalid",
				})
			}
		}

		for _, ref := range comp.ExternalRefs {
			if strings.TrimSpace(ref.Type) == "" || len(ref.Type) > 1024 || len(ref.URL) > 8192 || !isValidURL(ref.URL) {
				issues = append(issues, ValidationIssue{
					Field: "component.externalReferences.url", Severity: "error",
					Message: fmt.Sprintf("external reference requires a type and valid URL: %q", ref.URL),
				})
			}
		}
		for _, property := range comp.Properties {
			if strings.TrimSpace(property.Name) == "" || len(property.Name) > 4096 || len(property.Value) > 64<<10 ||
				strings.ContainsAny(property.Name+property.Value, "\x00") {
				issues = append(issues, ValidationIssue{Field: "component.properties", Severity: "error", Message: "component property is invalid or too large"})
			}
		}

		if comp.ModelCard != nil {
			if len(comp.ModelCard.BaseModel) > 4096 || len(comp.ModelCard.ModelFamily) > 4096 ||
				len(comp.ModelCard.Parameters) > 1024 || len(comp.ModelCard.TrustLabels) > 1024 ||
				len(comp.ModelCard.Adapters) > 10000 {
				issues = append(issues, ValidationIssue{Field: "component.modelCard", Severity: "error", Message: "model-card fields exceed supported limits"})
			}
			if comp.ModelCard.PromotionStatus != "" && !isValidPromotionStatus(comp.ModelCard.PromotionStatus) {
				issues = append(issues, ValidationIssue{
					Field: "component.modelCard.promotion_status", Severity: "error", Message: "invalid promotion status",
				})
			}
			for _, label := range comp.ModelCard.TrustLabels {
				if strings.TrimSpace(label) == "" || len(label) > 256 {
					issues = append(issues, ValidationIssue{
						Field: "component.modelCard.trust_labels", Severity: "error", Message: "trust label is empty or too long",
					})
				}
			}
			if quant := comp.ModelCard.Quantization; quant != nil {
				if strings.TrimSpace(quant.Method) == "" || len(quant.Method) > 1024 ||
					math.IsNaN(quant.BitsPerWeight) || math.IsInf(quant.BitsPerWeight, 0) || quant.BitsPerWeight < 0 || quant.BitsPerWeight > 64 ||
					len(quant.BaseModelHash) > 1024 || len(quant.QuantizedBy) > 4096 ||
					(quant.QuantizedAt != "" && !isValidRFC3339(quant.QuantizedAt)) {
					issues = append(issues, ValidationIssue{
						Field: "component.modelCard.quantization", Severity: "error", Message: "quantization fields are invalid",
					})
				}
			}
		}

		for _, ev := range comp.Evidence {
			issues = append(issues, validateEvidenceItem(ev)...)
		}
	}

	if len(bom.Dependencies) > 100000 {
		issues = append(issues, ValidationIssue{Field: "dependencies", Severity: "error", Message: "dependency count exceeds supported limits"})
	}
	dependencyRefs := make(map[string]struct{}, len(bom.Dependencies))
	for _, dependency := range bom.Dependencies {
		if _, exists := refs[dependency.Ref]; dependency.Ref == "" || !exists || len(dependency.DependsOn) > 100000 {
			issues = append(issues, ValidationIssue{Field: "dependencies.ref", Severity: "error", Message: "dependency references an unknown component"})
			continue
		}
		if _, duplicate := dependencyRefs[dependency.Ref]; duplicate {
			issues = append(issues, ValidationIssue{Field: "dependencies.ref", Severity: "error", Message: "duplicate dependency record"})
		}
		dependencyRefs[dependency.Ref] = struct{}{}
		seenTargets := make(map[string]struct{}, len(dependency.DependsOn))
		for _, target := range dependency.DependsOn {
			_, exists := refs[target]
			_, duplicate := seenTargets[target]
			if !exists || target == dependency.Ref || duplicate {
				issues = append(issues, ValidationIssue{Field: "dependencies.dependsOn", Severity: "error", Message: "dependency target is unknown, duplicated, or self-referential"})
			}
			seenTargets[target] = struct{}{}
		}
	}

	return issues
}

// validateEvidenceItem checks a single evidence item.
func validateEvidenceItem(ev EvidenceItem) []ValidationIssue {
	var issues []ValidationIssue

	if ev.Type == "" {
		issues = append(issues, ValidationIssue{
			Field: "evidence.type", Severity: "error",
			Message: "evidence item has empty type",
		})
	} else {
		validTypes := map[string]bool{"scan": true, "test": true, "attestation": true, "review": true}
		if !validTypes[ev.Type] {
			issues = append(issues, ValidationIssue{
				Field: "evidence.type", Severity: "error",
				Message: fmt.Sprintf("unknown evidence type: %q", ev.Type),
			})
		}
	}

	if strings.TrimSpace(ev.Source) == "" || len(ev.Source) > 1024 {
		issues = append(issues, ValidationIssue{
			Field: "evidence.source", Severity: "error",
			Message: "evidence source is required and must not exceed 1024 bytes",
		})
	}

	if ev.Result == "" {
		issues = append(issues, ValidationIssue{
			Field: "evidence.result", Severity: "error",
			Message: "evidence result is required",
		})
	} else {
		validResults := map[string]bool{"pass": true, "fail": true, "warning": true, "info": true}
		if !validResults[ev.Result] {
			issues = append(issues, ValidationIssue{
				Field: "evidence.result", Severity: "error",
				Message: fmt.Sprintf("invalid evidence result: %q; expected pass, fail, warning, or info", ev.Result),
			})
		}
	}

	if ev.Timestamp == "" || !isValidRFC3339(ev.Timestamp) {
		issues = append(issues, ValidationIssue{
			Field: "evidence.timestamp", Severity: "error",
			Message: fmt.Sprintf("evidence timestamp is required and must be valid RFC3339: %q", ev.Timestamp),
		})
	}
	if len(ev.Description) > 64<<10 || len(ev.Details) > 1024 {
		issues = append(issues, ValidationIssue{
			Field: "evidence", Severity: "error",
			Message: "evidence description or details exceed supported limits",
		})
	}
	for key, value := range ev.Details {
		if strings.TrimSpace(key) == "" || len(key) > 1024 || len(value) > 64<<10 ||
			strings.ContainsAny(key+value, "\x00") {
			issues = append(issues, ValidationIssue{
				Field: "evidence.details", Severity: "error",
				Message: "evidence detail keys and values must be bounded and free of NUL characters",
			})
			break
		}
	}
	if ev.Hash != "" {
		decoded, err := hex.DecodeString(ev.Hash)
		if err != nil || len(decoded) != 32 {
			issues = append(issues, ValidationIssue{
				Field: "evidence.hash", Severity: "error",
				Message: "evidence hash must be a SHA-256 digest",
			})
		}
	}

	return issues
}

// ---------- helpers ----------

func isValidURL(s string) bool {
	u, err := url.Parse(s)
	if err != nil || u.Host == "" || u.User != nil || u.Fragment != "" {
		return false
	}
	if u.Scheme == "https" {
		return true
	}
	if u.Scheme != "http" {
		return false
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func isValidRFC3339(s string) bool {
	_, err := time.Parse(time.RFC3339, s)
	return err == nil
}

func isValidPromotionStatus(s string) bool {
	valid := map[string]bool{
		"quarantine": true,
		"pending":    true,
		"approved":   true,
		"rejected":   true,
		"revoked":    true,
	}
	return valid[strings.ToLower(s)]
}

// Common SPDX license identifiers (not exhaustive, covers the most common ones).
var knownSPDX = map[string]bool{
	"Apache-2.0": true, "MIT": true, "GPL-2.0-only": true, "GPL-2.0-or-later": true,
	"GPL-3.0-only": true, "GPL-3.0-or-later": true, "BSD-2-Clause": true, "BSD-3-Clause": true,
	"ISC": true, "MPL-2.0": true, "LGPL-2.1-only": true, "LGPL-2.1-or-later": true,
	"LGPL-3.0-only": true, "LGPL-3.0-or-later": true, "AGPL-3.0-only": true,
	"Unlicense": true, "CC0-1.0": true, "CC-BY-4.0": true, "CC-BY-SA-4.0": true,
	"CC-BY-NC-4.0": true, "CC-BY-NC-SA-4.0": true, "WTFPL": true, "Zlib": true,
	"0BSD": true, "ECL-2.0": true, "AFL-3.0": true, "Artistic-2.0": true,
	"BSL-1.0": true, "PostgreSQL": true, "OFL-1.1": true, "EUPL-1.2": true,
	// Llama-specific community licenses
	"Llama-2": true, "Llama-3": true, "Llama-3.1": true, "Gemma": true,
}

var reSPDXLike = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9.\-]+$`)

func isKnownSPDX(s string) bool {
	if knownSPDX[s] {
		return true
	}
	// Accept anything that looks like a valid SPDX expression
	return reSPDXLike.MatchString(s)
}

// deduplicateStrings removes duplicates and sorts.
func deduplicateStrings(ss []string) []string {
	if len(ss) == 0 {
		return ss
	}
	seen := map[string]bool{}
	var out []string
	for _, s := range ss {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
