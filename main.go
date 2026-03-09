package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync/atomic"

	"gopkg.in/yaml.v3"
)

// ---------- metrics ----------

var (
	metricGenerated  atomic.Int64
	metricVerified   atomic.Int64
	metricDiffs      atomic.Int64
	metricHTTPReqs   atomic.Int64
)

// ---------- config ----------

// ServiceConfig is the top-level configuration.
type ServiceConfig struct {
	Version        int      `yaml:"version"`
	RequiredFields []string `yaml:"required_fields"`
	Daemon         struct {
		BindAddr string `yaml:"bind_addr"`
	} `yaml:"daemon"`
}

// ---------- main ----------

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "generate":
		cmdGenerate()
	case "verify":
		cmdVerify()
	case "diff":
		cmdDiff()
	case "attest":
		cmdAttest()
	case "evaluate":
		cmdEvaluate()
	case "serve":
		cmdServe()
	case "keygen":
		cmdKeygen()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `ai-model-bom — AI/ML BOM generator, verifier, and attestation companion

Usage:
  ai-model-bom generate   PATH [-meta FILE] [-out FILE]    Generate BOM from model
  ai-model-bom verify     FILE                              Verify signed BOM
  ai-model-bom diff       OLD NEW                           Compare two BOMs
  ai-model-bom attest     BOM [-subject S] [-predicate P]   Create signed attestation
  ai-model-bom evaluate   BOM                               Check deployment readiness
  ai-model-bom serve      [-config FILE]                    Run as HTTP daemon
  ai-model-bom keygen     [-out PREFIX]                     Generate Ed25519 keypair

Environment:
  AIMBOM_CONFIG     Path to config YAML (default: policies/default-policy.yaml)
  SERVICE_TOKEN     Bearer token for protected endpoints
  SIGNING_KEY       Base64-encoded Ed25519 private key
`)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func loadSigningKey() ed25519.PrivateKey {
	keyStr := os.Getenv("SIGNING_KEY")
	if keyStr == "" {
		return nil
	}
	data, err := base64.StdEncoding.DecodeString(keyStr)
	if err != nil || len(data) != ed25519.PrivateKeySize {
		return nil
	}
	return ed25519.PrivateKey(data)
}

func loadConfig() *ServiceConfig {
	path := envOr("AIMBOM_CONFIG", "policies/default-policy.yaml")
	for i, arg := range os.Args[2:] {
		if arg == "-config" && i+1 < len(os.Args[2:])-1 {
			path = os.Args[i+3]
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return &ServiceConfig{Version: 1}
	}

	var cfg ServiceConfig
	yaml.Unmarshal(data, &cfg)
	return &cfg
}

// ---------- CLI commands ----------

func cmdGenerate() {
	if len(os.Args) < 3 {
		log.Fatal("usage: ai-model-bom generate PATH [-meta FILE] [-out FILE]")
	}

	modelPath := os.Args[2]
	var meta *ModelMetadata
	outFile := ""

	for i, arg := range os.Args[3:] {
		switch arg {
		case "-meta":
			if i+1 < len(os.Args[3:])-1 {
				metaPath := os.Args[i+4]
				data, err := os.ReadFile(metaPath)
				if err != nil {
					log.Fatalf("read metadata: %v", err)
				}
				meta = &ModelMetadata{}
				yaml.Unmarshal(data, meta)
			}
		case "-out":
			if i+1 < len(os.Args[3:])-1 {
				outFile = os.Args[i+4]
			}
		}
	}

	bom, err := GenerateBOM(modelPath, meta)
	if err != nil {
		log.Fatalf("generate BOM: %v", err)
	}

	privKey := loadSigningKey()

	var output interface{}
	if privKey != nil {
		sb := SignBOM(bom, privKey, "ai-model-bom")
		output = sb
	} else {
		output = bom
	}

	metricGenerated.Add(1)

	if outFile != "" {
		data, _ := json.MarshalIndent(output, "", "  ")
		if err := os.WriteFile(outFile, data, 0o644); err != nil {
			log.Fatalf("write output: %v", err)
		}
		fmt.Printf("BOM written to %s (%d components)\n", outFile, len(bom.Components))
	} else {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(output)
	}
}

func cmdVerify() {
	if len(os.Args) < 3 {
		log.Fatal("usage: ai-model-bom verify FILE")
	}

	data, err := os.ReadFile(os.Args[2])
	if err != nil {
		log.Fatalf("read BOM: %v", err)
	}

	var sb SignedBOM
	if err := json.Unmarshal(data, &sb); err != nil {
		log.Fatalf("parse signed BOM: %v", err)
	}

	if sb.Signature == "" {
		fmt.Println("BOM is not signed")
		os.Exit(1)
	}

	// Load public key
	pubKeyStr := os.Getenv("VERIFY_KEY")
	if pubKeyStr == "" {
		log.Fatal("set VERIFY_KEY env to base64 public key")
	}
	pubKeyData, err := base64.StdEncoding.DecodeString(pubKeyStr)
	if err != nil {
		log.Fatalf("decode public key: %v", err)
	}

	valid, reason := VerifySignedBOM(sb, ed25519.PublicKey(pubKeyData))
	metricVerified.Add(1)

	if valid {
		fmt.Printf("VALID: BOM signature verified (signed_at=%s signed_by=%s)\n", sb.SignedAt, sb.SignedBy)
		fmt.Printf("  components: %d\n", len(sb.BOM.Components))
	} else {
		fmt.Printf("INVALID: %s\n", reason)
		os.Exit(2)
	}
}

func cmdDiff() {
	if len(os.Args) < 4 {
		log.Fatal("usage: ai-model-bom diff OLD NEW")
	}

	oldBOM := loadBOMFile(os.Args[2])
	newBOM := loadBOMFile(os.Args[3])

	result := DiffBOMs(oldBOM, newBOM)
	metricDiffs.Add(1)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(result)

	if result.TrustRelevant {
		os.Exit(1)
	}
}

func cmdAttest() {
	if len(os.Args) < 3 {
		log.Fatal("usage: ai-model-bom attest BOM [-subject S] [-predicate P]")
	}

	bom := loadBOMFile(os.Args[2])
	subject := ""
	predicate := "deployment-ready"

	for i, arg := range os.Args[3:] {
		switch arg {
		case "-subject":
			if i+1 < len(os.Args[3:])-1 {
				subject = os.Args[i+4]
			}
		case "-predicate":
			if i+1 < len(os.Args[3:])-1 {
				predicate = os.Args[i+4]
			}
		}
	}

	if subject == "" && len(bom.Components) > 0 {
		subject = bom.Components[0].Name
	}

	// Collect evidence from BOM components
	var evidence []EvidenceItem
	for _, comp := range bom.Components {
		evidence = append(evidence, comp.Evidence...)
	}

	att := CreateAttestation(subject, predicate, "ai-model-bom", evidence)

	privKey := loadSigningKey()
	if privKey != nil {
		SignAttestation(&att, privKey)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(att)
}

func cmdEvaluate() {
	if len(os.Args) < 3 {
		log.Fatal("usage: ai-model-bom evaluate BOM")
	}

	bom := loadBOMFile(os.Args[2])
	cfg := loadConfig()

	fields := cfg.RequiredFields
	if len(fields) == 0 {
		fields = []string{"hash", "license", "model_card"}
	}

	ready, missing := EvaluateReadiness(bom, fields)

	if ready {
		fmt.Println("READY: all required fields present")
	} else {
		fmt.Println("NOT READY: missing required fields:")
		for _, m := range missing {
			fmt.Printf("  - %s\n", m)
		}
		os.Exit(2)
	}
}

// ---------- HTTP daemon ----------

func cmdServe() {
	cfg := loadConfig()
	privKey := loadSigningKey()
	token := os.Getenv("SERVICE_TOKEN")

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	mux.HandleFunc("/v1/generate", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Path     string         `json:"path"`
			Metadata *ModelMetadata `json:"metadata,omitempty"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
			return
		}

		bom, err := GenerateBOM(req.Path, req.Metadata)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadRequest)
			return
		}

		metricGenerated.Add(1)

		resp := map[string]interface{}{"bom": bom}
		if privKey != nil {
			resp["signed_bom"] = SignBOM(bom, privKey, "ai-model-bom")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("/v1/verify", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var sb SignedBOM
		if err := json.NewDecoder(io.LimitReader(r.Body, 10<<20)).Decode(&sb); err != nil {
			http.Error(w, `{"error":"invalid signed BOM"}`, http.StatusBadRequest)
			return
		}

		metricVerified.Add(1)

		if privKey != nil {
			pubKey := privKey.Public().(ed25519.PublicKey)
			valid, reason := VerifySignedBOM(sb, pubKey)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"valid": valid, "reason": reason,
			})
		} else {
			http.Error(w, `{"error":"no signing key configured for verification"}`, http.StatusServiceUnavailable)
		}
	})

	mux.HandleFunc("/v1/diff", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Old *ModelBOM `json:"old"`
			New *ModelBOM `json:"new"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 20<<20)).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
			return
		}

		if req.Old == nil || req.New == nil {
			http.Error(w, `{"error":"both old and new BOMs required"}`, http.StatusBadRequest)
			return
		}

		result := DiffBOMs(req.Old, req.New)
		metricDiffs.Add(1)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	})

	mux.HandleFunc("/v1/evaluate", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var bom ModelBOM
		if err := json.NewDecoder(io.LimitReader(r.Body, 10<<20)).Decode(&bom); err != nil {
			http.Error(w, `{"error":"invalid BOM"}`, http.StatusBadRequest)
			return
		}

		fields := cfg.RequiredFields
		if len(fields) == 0 {
			fields = []string{"hash", "license", "model_card"}
		}

		ready, missing := EvaluateReadiness(&bom, fields)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ready":   ready,
			"missing": missing,
		})
	})

	mux.HandleFunc("/v1/attest", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !checkToken(r, token) {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}

		var req struct {
			Subject   string         `json:"subject"`
			Predicate string         `json:"predicate"`
			Evidence  []EvidenceItem `json:"evidence"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request"}`, http.StatusBadRequest)
			return
		}

		att := CreateAttestation(req.Subject, req.Predicate, "ai-model-bom", req.Evidence)
		if privKey != nil {
			SignAttestation(&att, privKey)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(att)
	})

	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int64{
			"generated_total":    metricGenerated.Load(),
			"verified_total":     metricVerified.Load(),
			"diffs_total":        metricDiffs.Load(),
			"http_requests_total": metricHTTPReqs.Load(),
		})
	})

	addr := cfg.Daemon.BindAddr
	if addr == "" {
		addr = "127.0.0.1:8515"
	}

	log.Printf("ai-model-bom serving on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func cmdKeygen() {
	prefix := "ai-model-bom"
	for i, arg := range os.Args[2:] {
		if arg == "-out" && i+1 < len(os.Args[2:])-1 {
			prefix = os.Args[i+3]
		}
	}

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		log.Fatalf("keygen failed: %v", err)
	}

	privB64 := base64.StdEncoding.EncodeToString(priv)
	pubB64 := base64.StdEncoding.EncodeToString(pub)

	os.WriteFile(prefix+".key", []byte(privB64+"\n"), 0o600)
	os.WriteFile(prefix+".pub", []byte(pubB64+"\n"), 0o644)

	fmt.Printf("Keys written: %s.key (private), %s.pub (public)\n", prefix, prefix)
}

// ---------- helpers ----------

func checkToken(r *http.Request, expected string) bool {
	if expected == "" {
		return true
	}
	auth := r.Header.Get("Authorization")
	provided := strings.TrimPrefix(auth, "Bearer ")
	return subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) == 1
}

func loadBOMFile(path string) *ModelBOM {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read BOM: %v", err)
	}

	// Try SignedBOM first
	var sb SignedBOM
	if err := json.Unmarshal(data, &sb); err == nil && sb.BOM != nil {
		return sb.BOM
	}

	var bom ModelBOM
	if err := json.Unmarshal(data, &bom); err != nil {
		log.Fatalf("parse BOM: %v", err)
	}
	return &bom
}
