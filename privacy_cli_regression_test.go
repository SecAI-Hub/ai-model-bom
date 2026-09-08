package main

import (
	"encoding/json"
	"flag"
	"strings"
	"testing"
)

func TestGenerateFlagOrderingHonorsEveryOption(t *testing.T) {
	for _, args := range [][]string{
		{"/models/a.gguf", "--out", "report.json", "--privacy-profile", "default"},
		{"--out", "report.json", "/models/a.gguf", "--privacy-profile=default"},
	} {
		fs := flag.NewFlagSet("fixture", flag.ContinueOnError)
		out := fs.String("out", "", "")
		privacy := fs.String("privacy-profile", "", "")
		if err := parseGenerateFlags(fs, args); err != nil {
			t.Fatal(err)
		}
		if *out != "report.json" || *privacy != "default" || fs.NArg() != 1 || fs.Arg(0) != "/models/a.gguf" {
			t.Fatalf("silently ignored options: %v", args)
		}
	}
	fs := flag.NewFlagSet("fixture", flag.ContinueOnError)
	if err := parseGenerateFlags(fs, []string{"one.gguf", "two.gguf"}); err == nil {
		t.Fatal("extra path silently ignored")
	}
}

func TestEvidencePrivacyCoversSourceKeysPathsAndAddresses(t *testing.T) {
	items := []EvidenceItem{{Source: "/srv/private/model.gguf", Description: "host=[2001:db8::1] file=file:///run/private/secret", Details: map[string]string{"person@example.com": "value", "other@example.com": "different"}}}
	out := RedactEvidence(items, DefaultPrivacyProfile())
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, sensitive := range []string{"/srv/private", "/run/private", "2001:db8::1", "person@example.com", "other@example.com"} {
		if strings.Contains(string(data), sensitive) {
			t.Fatalf("evidence privacy missed %s", sensitive)
		}
	}
	if len(out[0].Details) != 2 {
		t.Fatal("redacted key collision silently discarded evidence")
	}
}
