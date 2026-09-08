package main

import (
	"fmt"
	"net"
	"regexp"
	"sort"
	"strings"
)

// PrivacyProfile controls what gets stripped from evidence before export.
type PrivacyProfile struct {
	StripEmails    bool `yaml:"strip_emails" json:"strip_emails"`
	StripPaths     bool `yaml:"strip_paths" json:"strip_paths"`
	StripHostnames bool `yaml:"strip_hostnames" json:"strip_hostnames"`
	StripUsernames bool `yaml:"strip_usernames" json:"strip_usernames"`
}

// DefaultPrivacyProfile returns a profile that strips common PII.
func DefaultPrivacyProfile() PrivacyProfile {
	return PrivacyProfile{
		StripEmails:    true,
		StripPaths:     true,
		StripHostnames: true,
		StripUsernames: true,
	}
}

var (
	reEmail    = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)
	reAbsPath  = regexp.MustCompile(`(?:^|[\s"':=])(/[^/\s"'][^\s"']*)`)
	reFileURI  = regexp.MustCompile(`(?i)\bfile://[^\s"'<>]+`)
	reAddress  = regexp.MustCompile(`(?:[0-9]{1,3}\.){3}[0-9]{1,3}|[0-9a-fA-F]*:[0-9a-fA-F:.]+(?:%[A-Za-z0-9_.-]+)?`)
	reWinPath  = regexp.MustCompile(`[A-Z]:\\[^\s"']+`)
	reHostname = regexp.MustCompile(`(?i)\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:local|internal|corp|lan|intra|priv)\b`)
	reUsername = regexp.MustCompile(`(?i)(?:user(?:name)?|reviewer|actor|author|admin)\s*[:=]\s*["']?([^\s"',]+)["']?`)
)

// RedactEvidence applies privacy controls to evidence items.
// Returns a new slice; originals are not modified.
func RedactEvidence(items []EvidenceItem, profile PrivacyProfile) []EvidenceItem {
	out := make([]EvidenceItem, len(items))
	for i, item := range items {
		out[i] = redactEvidenceItem(item, profile)
	}
	return out
}

func redactEvidenceItem(item EvidenceItem, profile PrivacyProfile) EvidenceItem {
	// Copy the item
	result := EvidenceItem{
		Type:        item.Type,
		Source:      redactString(item.Source, profile),
		Timestamp:   item.Timestamp,
		Result:      item.Result,
		Description: redactString(item.Description, profile),
		Hash:        item.Hash,
	}

	if item.Details != nil {
		result.Details = make(map[string]string, len(item.Details))
		keys := make([]string, 0, len(item.Details))
		for k := range item.Details {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			v := item.Details[k]
			key := redactString(k, profile)
			base := key
			for suffix := 2; ; suffix++ {
				if _, exists := result.Details[key]; !exists {
					break
				}
				key = fmt.Sprintf("%s#%d", base, suffix)
			}
			// Strip detail keys that are inherently sensitive
			if profile.StripUsernames && isSensitiveKey(k) {
				result.Details[key] = "[REDACTED]"
			} else {
				result.Details[key] = redactString(v, profile)
			}
		}
	}

	return result
}

func redactString(s string, profile PrivacyProfile) string {
	if s == "" {
		return s
	}

	if profile.StripEmails {
		s = reEmail.ReplaceAllString(s, "[REDACTED:email]")
	}
	if profile.StripPaths {
		s = reFileURI.ReplaceAllString(s, "[REDACTED:path]")
		s = reAbsPath.ReplaceAllStringFunc(s, func(match string) string {
			// Preserve the leading whitespace/delimiter
			for i, c := range match {
				if c == '/' {
					return match[:i] + "[REDACTED:path]"
				}
			}
			return "[REDACTED:path]"
		})
		s = reWinPath.ReplaceAllString(s, "[REDACTED:path]")
	}
	if profile.StripHostnames {
		s = reHostname.ReplaceAllString(s, "[REDACTED:hostname]")
		s = reAddress.ReplaceAllStringFunc(s, func(candidate string) string {
			// A preceding label can contribute hexadecimal letters to a match;
			// try colon-delimited suffixes without accepting invalid addresses.
			for value := candidate; value != ""; {
				if net.ParseIP(strings.SplitN(value, "%", 2)[0]) != nil {
					return "[REDACTED:address]"
				}
				i := strings.IndexByte(value, ':')
				if i < 0 {
					break
				}
				value = value[i+1:]
			}
			return candidate
		})
	}
	if profile.StripUsernames {
		s = reUsername.ReplaceAllStringFunc(s, func(match string) string {
			parts := strings.SplitN(match, ":", 2)
			if len(parts) == 2 {
				return parts[0] + ": [REDACTED:username]"
			}
			parts = strings.SplitN(match, "=", 2)
			if len(parts) == 2 {
				return parts[0] + "= [REDACTED:username]"
			}
			return "[REDACTED:username]"
		})
	}

	return s
}

func isSensitiveKey(key string) bool {
	lower := strings.ToLower(key)
	sensitiveKeys := []string{"reviewer", "actor", "author", "user", "username", "admin", "email", "operator"}
	for _, sk := range sensitiveKeys {
		if lower == sk {
			return true
		}
	}
	return false
}

// RedactBOM applies privacy controls to all evidence in a BOM.
func RedactBOM(bom *ModelBOM, profile PrivacyProfile) *ModelBOM {
	// Deep copy components with redacted evidence
	result := *bom
	result.Components = make([]Component, len(bom.Components))
	for i, comp := range bom.Components {
		result.Components[i] = comp
		if len(comp.Evidence) > 0 {
			result.Components[i].Evidence = RedactEvidence(comp.Evidence, profile)
		}
	}
	return &result
}
