package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// ---------- metrics ----------

var (
	metricGenerated atomic.Int64
	metricVerified  atomic.Int64
	metricDiffs     atomic.Int64
	metricHTTPReqs  atomic.Int64
)

// ---------- config ----------

// ServiceConfig is the top-level configuration.
type ServiceConfig struct {
	Version                  int      `yaml:"version"`
	RequiredFields           []string `yaml:"required_fields"`
	RequireSignedTrustInputs *bool    `yaml:"require_signed_trust_inputs,omitempty"`
	Daemon                   struct {
		BindAddr          string   `yaml:"bind_addr"`
		AllowedPaths      []string `yaml:"allowed_paths"`
		ReadTimeout       int      `yaml:"read_timeout_seconds"`
		WriteTimeout      int      `yaml:"write_timeout_seconds"`
		RequestsPerMinute int      `yaml:"requests_per_minute"`
	} `yaml:"daemon"`
	PrivacyProfile *PrivacyProfile `yaml:"privacy_profile,omitempty"`
	allowedRoots   []*pinnedAllowedRoot
}

type pinnedAllowedRoot struct {
	canonicalPath string
	root          *os.Root
}

// ---------- main ----------

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "generate":
		cmdGenerate(os.Args[2:])
	case "verify":
		cmdVerify(os.Args[2:])
	case "diff":
		cmdDiff(os.Args[2:])
	case "attest":
		cmdAttest(os.Args[2:])
	case "evaluate":
		cmdEvaluate(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "keygen":
		cmdKeygen(os.Args[2:])
	case "help", "--help", "-h":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `ai-model-bom — AI/ML BOM generator, verifier, and attestation companion

Usage:
  ai-model-bom <command> [flags]

Commands:
  generate   Generate BOM from a model file or directory
  verify     Verify a signed BOM
  diff       Compare two BOMs
  attest     Create a signed attestation
  evaluate   Check deployment readiness
  serve      Run as HTTP daemon
  keygen     Generate Ed25519 keypair

Run "ai-model-bom <command> --help" for command-specific flags.

Environment:
  AIMBOM_CONFIG   Path to config YAML (default: policies/default-policy.yaml)
	  SERVICE_TOKEN   Bearer token for protected endpoints
	  SERVICE_TOKEN_PATH  Path to an owner-only bearer-token file
	  SIGNING_SERVICE_TOKEN  Separate bearer token for endpoints that use the signing key
	  SIGNING_SERVICE_TOKEN_PATH  Path to an owner-only signing-authority token file
  SIGNING_KEY     Base64-encoded Ed25519 private key
  SIGNING_KEY_PATH  Path to an owner-only base64 private-key file
  VERIFY_KEY      Base64-encoded Ed25519 public key (for verify command)
  VERIFY_KEY_PATH  Path to a base64 public-key file
`)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func loadSigningKey() (ed25519.PrivateKey, error) {
	encoded, err := loadSecret("SIGNING_KEY", "SIGNING_KEY_PATH", false)
	if err != nil || encoded == "" {
		return nil, err
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, errors.New("SIGNING_KEY is not valid base64")
	}
	if len(data) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("SIGNING_KEY must decode to %d bytes", ed25519.PrivateKeySize)
	}
	return ed25519.PrivateKey(data), nil
}

func loadVerifyKey() (ed25519.PublicKey, error) {
	encoded, err := loadSecret("VERIFY_KEY", "VERIFY_KEY_PATH", true)
	if err != nil || encoded == "" {
		return nil, err
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, errors.New("VERIFY_KEY is not valid base64")
	}
	if len(data) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("VERIFY_KEY must decode to %d bytes", ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(data), nil
}

func loadServiceToken() (string, error) {
	token, err := loadSecret("SERVICE_TOKEN", "SERVICE_TOKEN_PATH", false)
	if err != nil {
		return "", err
	}
	if !validServiceToken(token) {
		return "", errors.New("SERVICE_TOKEN must contain at least 32 bytes without whitespace or control characters")
	}
	return token, nil
}

func loadSigningServiceToken() (string, error) {
	token, err := loadSecret("SIGNING_SERVICE_TOKEN", "SIGNING_SERVICE_TOKEN_PATH", false)
	if err != nil || token == "" {
		return token, err
	}
	if !validServiceToken(token) {
		return "", errors.New("SIGNING_SERVICE_TOKEN must contain at least 32 bytes without whitespace or control characters")
	}
	return token, nil
}

func validServiceToken(token string) bool {
	if len(token) < 32 {
		return false
	}
	for i := 0; i < len(token); i++ {
		if token[i] <= 0x20 || token[i] == 0x7f {
			return false
		}
	}
	return true
}

func loadSecret(envName, pathEnvName string, publicFile bool) (string, error) {
	direct := strings.TrimSpace(os.Getenv(envName))
	path := strings.TrimSpace(os.Getenv(pathEnvName))
	if direct != "" && path != "" {
		return "", fmt.Errorf("set only one of %s or %s", envName, pathEnvName)
	}
	if path == "" {
		return direct, nil
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", pathEnvName, err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("%s must name a regular file", pathEnvName)
	}
	if !publicFile && info.Mode().Perm()&0o077 != 0 {
		return "", fmt.Errorf("%s must not be accessible by group or other users", pathEnvName)
	}
	data, err := readRegularFile(path, maxSecretBytes)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", pathEnvName, err)
	}
	value := strings.TrimSpace(string(data))
	if value == "" {
		return "", fmt.Errorf("%s is empty", pathEnvName)
	}
	return value, nil
}

func loadConfig(configPath string) (*ServiceConfig, error) {
	if configPath == "" {
		configPath = envOr("AIMBOM_CONFIG", "policies/default-policy.yaml")
	}

	data, err := readRegularFile(configPath, maxConfigBytes)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg ServiceConfig
	if err := decodeStrictYAML(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if err := validateConfig(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func validateConfig(cfg *ServiceConfig) error {
	if cfg == nil || cfg.Version != 1 {
		return errors.New("config version must be 1")
	}
	validFields := map[string]bool{
		"hash": true, "license": true, "model_card": true, "evidence": true,
		"trust_labels": true, "quantization": true, "source": true,
		"promotion_status": true,
	}
	seen := make(map[string]bool)
	for _, field := range cfg.RequiredFields {
		if !validFields[field] {
			return fmt.Errorf("unsupported required field %q", field)
		}
		if seen[field] {
			return fmt.Errorf("duplicate required field %q", field)
		}
		seen[field] = true
	}
	if cfg.Daemon.ReadTimeout < 0 || cfg.Daemon.ReadTimeout > 300 ||
		cfg.Daemon.WriteTimeout < 0 || cfg.Daemon.WriteTimeout > 600 {
		return errors.New("daemon timeout is outside the supported range")
	}
	if cfg.Daemon.RequestsPerMinute < 0 || cfg.Daemon.RequestsPerMinute > 100000 {
		return errors.New("daemon requests_per_minute is outside the supported range")
	}
	if cfg.Daemon.BindAddr != "" {
		host, port, err := net.SplitHostPort(cfg.Daemon.BindAddr)
		if err != nil || strings.TrimSpace(port) == "" {
			return fmt.Errorf("invalid daemon bind_addr %q", cfg.Daemon.BindAddr)
		}
		if host == "" {
			return errors.New("daemon bind_addr must specify an explicit host")
		}
	}
	return nil
}

// ---------- CLI commands ----------

func cmdGenerate(args []string) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	metaFile := fs.String("meta", "", "Path to metadata YAML file")
	outFile := fs.String("out", "", "Output file path (default: stdout)")
	evidenceFile := fs.String("evidence", "", "Path to evidence JSON file to attach")
	evidenceComp := fs.String("evidence-component", "", "Component bom-ref or name to attach evidence to")
	privacyFlag := fs.String("privacy-profile", "", "Privacy profile: none, default, or path to YAML")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom generate PATH [flags]\n\nFlags:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	if fs.NArg() < 1 {
		fs.Usage()
		os.Exit(1)
	}

	modelPath := fs.Arg(0)

	// Load metadata if provided
	var meta *ModelMetadata
	if *metaFile != "" {
		data, err := readRegularFile(*metaFile, maxMetadataBytes)
		if err != nil {
			log.Fatalf("read metadata: %v", err)
		}
		meta = &ModelMetadata{}
		if err := decodeStrictYAML(data, meta); err != nil {
			log.Fatalf("parse metadata: %v", err)
		}
		// Validate metadata
		if issues := ValidateMetadata(meta); len(issues) > 0 {
			for _, iss := range issues {
				log.Printf("metadata %s: %s: %s", iss.Severity, iss.Field, iss.Message)
			}
			hasError := false
			for _, iss := range issues {
				if iss.Severity == "error" {
					hasError = true
				}
			}
			if hasError {
				log.Fatal("metadata validation failed with errors")
			}
		}
	}

	bom, err := GenerateBOM(modelPath, meta)
	if err != nil {
		log.Fatalf("generate BOM: %v", err)
	}

	// Attach evidence if provided
	if *evidenceFile != "" {
		items, err := ImportEvidence(*evidenceFile)
		if err != nil {
			log.Fatalf("import evidence: %v", err)
		}
		targetRef := *evidenceComp
		if targetRef == "" && len(bom.Components) > 0 {
			targetRef = bom.Components[0].BOMRef
		}
		if !AttachEvidence(bom, targetRef, items) {
			log.Fatalf("component not found: %s", targetRef)
		}
	}

	// Apply privacy redaction if requested
	if *privacyFlag != "" {
		profile, err := resolvePrivacyProfile(*privacyFlag)
		if err != nil {
			log.Fatalf("load privacy profile: %v", err)
		}
		bom = RedactBOM(bom, profile)
	}

	// Validate generated BOM
	hasValidationError := false
	if issues := ValidateBOM(bom); len(issues) > 0 {
		for _, iss := range issues {
			if iss.Severity == "error" {
				log.Printf("BOM validation %s: %s: %s", iss.Severity, iss.Field, iss.Message)
				hasValidationError = true
			}
		}
	}
	if hasValidationError {
		log.Fatal("generated BOM failed validation")
	}

	privKey, err := loadSigningKey()
	if err != nil {
		log.Fatalf("load signing key: %v", err)
	}

	var output interface{}
	if privKey != nil {
		signed, err := SignBOM(bom, privKey, "ai-model-bom")
		if err != nil {
			log.Fatalf("sign BOM: %v", err)
		}
		output = signed
	} else {
		output = bom
	}

	metricGenerated.Add(1)

	if *outFile != "" {
		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			log.Fatalf("encode output: %v", err)
		}
		data = append(data, '\n')
		if err := writeFileAtomic(*outFile, data, 0o644); err != nil {
			log.Fatalf("write output: %v", err)
		}
		fmt.Printf("BOM written to %s (%d components)\n", *outFile, len(bom.Components))
	} else {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(output); err != nil {
			log.Fatalf("encode output: %v", err)
		}
	}
}

func cmdVerify(args []string) {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom verify FILE\n\nVerify a signed BOM's integrity and signature.\nSet VERIFY_KEY env to base64-encoded Ed25519 public key.\n")
	}
	fs.Parse(args)

	if fs.NArg() < 1 {
		fs.Usage()
		os.Exit(1)
	}

	data, err := readRegularFile(fs.Arg(0), maxBOMBytes)
	if err != nil {
		log.Fatalf("read BOM: %v", err)
	}

	var sb SignedBOM
	if err := decodeStrictJSON(data, &sb); err != nil {
		log.Fatalf("parse signed BOM: %v", err)
	}

	if sb.Signature == "" {
		fmt.Println("BOM is not signed")
		os.Exit(1)
	}

	pubKey, err := loadVerifyKey()
	if err != nil {
		log.Fatalf("load verification key: %v", err)
	}
	if pubKey == nil {
		log.Fatal("set VERIFY_KEY or VERIFY_KEY_PATH")
	}

	valid, reason := VerifySignedBOM(sb, pubKey)
	metricVerified.Add(1)

	if valid {
		fmt.Printf("VALID: BOM signature verified (signed_at=%s signed_by=%s)\n", sb.SignedAt, sb.SignedBy)
		fmt.Printf("  components: %d\n", len(sb.BOM.Components))
	} else {
		fmt.Printf("INVALID: %s\n", reason)
		os.Exit(2)
	}
}

func cmdDiff(args []string) {
	fs := flag.NewFlagSet("diff", flag.ExitOnError)
	allowUnsigned := fs.Bool("allow-unsigned", false, "allow unsigned BOMs for structural comparison only")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom diff OLD NEW\n\nCompare two BOMs and report trust-relevant changes.\n")
	}
	fs.Parse(args)

	if fs.NArg() < 2 {
		fs.Usage()
		os.Exit(1)
	}

	oldBOM, err := loadBOMFileWithPolicy(fs.Arg(0), *allowUnsigned)
	if err != nil {
		log.Fatalf("load old BOM: %v", err)
	}
	newBOM, err := loadBOMFileWithPolicy(fs.Arg(1), *allowUnsigned)
	if err != nil {
		log.Fatalf("load new BOM: %v", err)
	}

	result := DiffBOMs(oldBOM, newBOM)
	metricDiffs.Add(1)

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		log.Fatalf("encode diff: %v", err)
	}

	if result.TrustRelevant {
		os.Exit(1)
	}
}

func cmdAttest(args []string) {
	fs := flag.NewFlagSet("attest", flag.ExitOnError)
	subject := fs.String("subject", "", "Subject (model name or hash)")
	predicate := fs.String("predicate", "deployment-ready", "Predicate being attested")
	allowUnsigned := fs.Bool("allow-unsigned", false, "explicitly sign evidence from an unsigned BOM")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom attest BOM [flags]\n\nCreate a signed attestation for a model BOM.\n\nFlags:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	if fs.NArg() < 1 {
		fs.Usage()
		os.Exit(1)
	}

	bom, err := loadBOMFileWithPolicy(fs.Arg(0), *allowUnsigned)
	if err != nil {
		log.Fatalf("load BOM: %v", err)
	}

	if *subject == "" && len(bom.Components) > 0 {
		*subject = bom.Components[0].Name
	}

	// Collect evidence from BOM components
	var evidence []EvidenceItem
	for _, comp := range bom.Components {
		evidence = append(evidence, comp.Evidence...)
	}

	att := CreateAttestation(*subject, *predicate, "ai-model-bom", evidence)

	privKey, err := loadSigningKey()
	if err != nil {
		log.Fatalf("load signing key: %v", err)
	}
	if privKey == nil {
		log.Fatal("attest requires SIGNING_KEY or SIGNING_KEY_PATH")
	}
	if err := SignAttestation(&att, privKey); err != nil {
		log.Fatalf("sign attestation: %v", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(att); err != nil {
		log.Fatalf("encode attestation: %v", err)
	}
}

func cmdEvaluate(args []string) {
	fs := flag.NewFlagSet("evaluate", flag.ExitOnError)
	configFile := fs.String("config", "", "Path to config YAML")
	allowUnsigned := fs.Bool("allow-unsigned", false, "allow an unsigned BOM for structural readiness evaluation")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom evaluate BOM [flags]\n\nCheck if a BOM meets deployment readiness criteria.\n\nFlags:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	if fs.NArg() < 1 {
		fs.Usage()
		os.Exit(1)
	}

	bom, err := loadBOMFileWithPolicy(fs.Arg(0), *allowUnsigned)
	if err != nil {
		log.Fatalf("load BOM: %v", err)
	}
	cfg, err := loadConfig(*configFile)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

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

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	configFile := fs.String("config", "", "Path to config YAML")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom serve [flags]\n\nRun as an HTTP daemon.\n\nFlags:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	cfg, err := loadConfig(*configFile)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	if len(cfg.Daemon.AllowedPaths) == 0 {
		log.Fatal("daemon.allowed_paths must contain at least one directory")
	}
	for _, allowed := range cfg.Daemon.AllowedPaths {
		info, err := os.Stat(allowed)
		if err != nil || !info.IsDir() {
			log.Fatalf("allowed path %q must be an existing directory", allowed)
		}
	}
	allowedRoots, err := openAllowedRoots(cfg.Daemon.AllowedPaths)
	if err != nil {
		log.Fatalf("open daemon allowlist: %v", err)
	}
	cfg.allowedRoots = allowedRoots
	defer closeAllowedRoots(allowedRoots)
	privKey, err := loadSigningKey()
	if err != nil {
		log.Fatalf("load signing key: %v", err)
	}
	verifyKey, err := loadVerifyKey()
	if err != nil {
		log.Fatalf("load verification key: %v", err)
	}
	if verifyKey == nil && privKey != nil {
		verifyKey = privKey.Public().(ed25519.PublicKey)
	}
	if requiresSignedTrustInputs(cfg) && verifyKey == nil {
		log.Fatal("require_signed_trust_inputs requires VERIFY_KEY/VERIFY_KEY_PATH or a signing key")
	}
	token, err := loadServiceToken()
	if err != nil {
		log.Fatalf("load service token: %v", err)
	}
	signingToken, err := loadSigningServiceToken()
	if err != nil {
		log.Fatalf("load signing service token: %v", err)
	}
	if signingToken != "" && subtle.ConstantTimeCompare([]byte(signingToken), []byte(token)) == 1 {
		log.Fatal("SIGNING_SERVICE_TOKEN must be distinct from SERVICE_TOKEN")
	}
	if privKey != nil && signingToken == "" {
		log.Printf("signing key is configured but signing HTTP endpoints are disabled until SIGNING_SERVICE_TOKEN or SIGNING_SERVICE_TOKEN_PATH is set")
	}

	mux := buildMuxWithTokens(cfg, privKey, verifyKey, token, signingToken)

	addr := cfg.Daemon.BindAddr
	if addr == "" {
		addr = "127.0.0.1:8515"
	}
	if !isLoopbackBind(addr) && os.Getenv("ALLOW_REMOTE_BIND") != "true" {
		log.Fatalf("remote bind %q requires ALLOW_REMOTE_BIND=true and a protected TLS/network boundary", addr)
	}

	readTimeout := cfg.Daemon.ReadTimeout
	if readTimeout <= 0 {
		readTimeout = 30
	}
	writeTimeout := cfg.Daemon.WriteTimeout
	if writeTimeout <= 0 {
		writeTimeout = 60
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           securityHeaders(auditHTTP(mux)),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       time.Duration(readTimeout) * time.Second,
		WriteTimeout:      time.Duration(writeTimeout) * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}

	log.Printf("ai-model-bom serving on %s", addr)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Printf("graceful shutdown failed: %v", err)
		}
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("serve: %v", err)
	}
}

// buildMux constructs the HTTP handler. Exported for testing.
func buildMux(cfg *ServiceConfig, privKey ed25519.PrivateKey, token string) *http.ServeMux {
	var verifyKey ed25519.PublicKey
	if len(privKey) == ed25519.PrivateKeySize {
		verifyKey = privKey.Public().(ed25519.PublicKey)
	}
	return buildMuxWithVerifyKey(cfg, privKey, verifyKey, token)
}

func buildMuxWithVerifyKey(cfg *ServiceConfig, privKey ed25519.PrivateKey, verifyKey ed25519.PublicKey, token string) *http.ServeMux {
	return buildMuxWithTokens(cfg, privKey, verifyKey, token, "")
}

// buildMuxWithTokens keeps trust-evaluation authority separate from signing
// authority. A bearer token that can verify/evaluate must not automatically be
// able to mint signed BOMs or arbitrary signed attestations.
func buildMuxWithTokens(cfg *ServiceConfig, privKey ed25519.PrivateKey, verifyKey ed25519.PublicKey, token, signingToken string) *http.ServeMux {
	mux := http.NewServeMux()
	if cfg != nil && cfg.allowedRoots == nil && len(cfg.Daemon.AllowedPaths) > 0 {
		// Tests and embedded callers do not pass through cmdServe. Pin any valid
		// roots at mux construction; production treats an error as fatal above.
		cfg.allowedRoots, _ = openAllowedRoots(cfg.Daemon.AllowedPaths)
	}
	rateLimit := 120
	if cfg != nil && cfg.Daemon.RequestsPerMinute > 0 {
		rateLimit = cfg.Daemon.RequestsPerMinute
	}
	limiter := &requestRateLimiter{limit: rateLimit, windowStart: time.Now()}
	concurrencySlots := make(chan struct{}, 16)

	// ---------- unauthenticated ----------

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// ---------- authenticated endpoints ----------

	mux.HandleFunc("/v1/generate", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		generationToken := token
		if privKey != nil {
			generationToken = signingToken
		}
		release, ok := beginProtectedRequest(w, r, generationToken, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()

		var req struct {
			Path     string         `json:"path"`
			Metadata *ModelMetadata `json:"metadata,omitempty"`
		}
		if err := decodeRequest(w, r, &req, 1<<20); err != nil {
			jsonError(w, "invalid request", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Path) == "" {
			jsonError(w, "path is required", http.StatusBadRequest)
			return
		}

		// Resolve against roots whose descriptors were pinned when the daemon
		// started. Later ancestor/symlink replacement cannot escape those roots.
		allowedRoot, relativePath, resolvedPath, ok := resolveAllowedTarget(req.Path, cfg.allowedRoots)
		if !ok {
			jsonError(w, "path not in allowed_paths", http.StatusForbidden)
			return
		}

		// Validate metadata
		if req.Metadata != nil {
			if issues := ValidateMetadata(req.Metadata); len(issues) > 0 {
				for _, iss := range issues {
					if iss.Severity == "error" {
						jsonError(w, "metadata validation failed", http.StatusBadRequest)
						return
					}
				}
			}
		}

		bom, err := GenerateBOMFromRoot(allowedRoot.root, relativePath, resolvedPath, req.Metadata)
		if err != nil {
			log.Printf("generate request failed: %v", err)
			jsonError(w, "BOM generation failed", http.StatusBadRequest)
			return
		}

		metricGenerated.Add(1)

		resp := map[string]interface{}{"bom": bom}
		if privKey != nil {
			signed, err := SignBOM(bom, privKey, "ai-model-bom")
			if err != nil {
				log.Printf("sign generated BOM: %v", err)
				jsonError(w, "BOM signing failed", http.StatusInternalServerError)
				return
			}
			resp["signed_bom"] = signed
		}

		writeJSON(w, http.StatusOK, resp)
	})

	mux.HandleFunc("/v1/verify", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		release, ok := beginProtectedRequest(w, r, token, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()

		var sb SignedBOM
		if err := decodeRequest(w, r, &sb, 10<<20); err != nil {
			jsonError(w, "invalid signed BOM", http.StatusBadRequest)
			return
		}

		metricVerified.Add(1)

		if verifyKey != nil {
			valid, reason := VerifySignedBOM(sb, verifyKey)
			writeJSON(w, http.StatusOK, map[string]interface{}{
				"valid": valid, "reason": reason,
			})
		} else {
			jsonError(w, "no verification key configured", http.StatusServiceUnavailable)
		}
	})

	mux.HandleFunc("/v1/diff", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		release, ok := beginProtectedRequest(w, r, token, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()

		var req struct {
			Old json.RawMessage `json:"old"`
			New json.RawMessage `json:"new"`
		}
		if err := decodeRequest(w, r, &req, 20<<20); err != nil {
			jsonError(w, "invalid request", http.StatusBadRequest)
			return
		}

		if len(req.Old) == 0 || len(req.New) == 0 {
			jsonError(w, "both old and new BOMs required", http.StatusBadRequest)
			return
		}
		requireSigned := requiresSignedTrustInputs(cfg)
		oldBOM, oldVerified, err := decodeTrustedBOM(req.Old, verifyKey, requireSigned)
		if err != nil {
			jsonError(w, "old BOM is not a valid trusted input", http.StatusBadRequest)
			return
		}
		newBOM, newVerified, err := decodeTrustedBOM(req.New, verifyKey, requireSigned)
		if err != nil {
			jsonError(w, "new BOM is not a valid trusted input", http.StatusBadRequest)
			return
		}

		result := DiffBOMs(oldBOM, newBOM)
		metricDiffs.Add(1)
		w.Header().Set("SecAI-Input-Verified", fmt.Sprintf("%t", oldVerified && newVerified))
		writeJSON(w, http.StatusOK, result)
	})

	mux.HandleFunc("/v1/evaluate", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		release, ok := beginProtectedRequest(w, r, token, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()

		var input json.RawMessage
		if err := decodeRequest(w, r, &input, 10<<20); err != nil {
			jsonError(w, "invalid BOM", http.StatusBadRequest)
			return
		}
		bom, verified, err := decodeTrustedBOM(input, verifyKey, requiresSignedTrustInputs(cfg))
		if err != nil {
			jsonError(w, "BOM is not a valid trusted input", http.StatusBadRequest)
			return
		}

		fields := cfg.RequiredFields
		if len(fields) == 0 {
			fields = []string{"hash", "license", "model_card"}
		}

		ready, missing := EvaluateReadiness(bom, fields)

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"ready": ready, "missing": missing, "input_verified": verified,
		})
	})

	mux.HandleFunc("/v1/attest", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodPost {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		release, ok := beginProtectedRequest(w, r, signingToken, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()
		if privKey == nil {
			jsonError(w, "no signing key configured", http.StatusServiceUnavailable)
			return
		}

		var req struct {
			Subject   string         `json:"subject"`
			Predicate string         `json:"predicate"`
			Evidence  []EvidenceItem `json:"evidence"`
		}
		if err := decodeRequest(w, r, &req, 1<<20); err != nil {
			jsonError(w, "invalid request", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Subject) == "" || len(req.Subject) > 1024 ||
			strings.TrimSpace(req.Predicate) == "" || len(req.Predicate) > 1024 ||
			len(req.Evidence) == 0 || len(req.Evidence) > 10000 {
			jsonError(w, "invalid attestation fields", http.StatusBadRequest)
			return
		}
		for _, evidence := range req.Evidence {
			for _, issue := range validateEvidenceItem(evidence) {
				if issue.Severity == "error" {
					jsonError(w, "invalid attestation evidence", http.StatusBadRequest)
					return
				}
			}
		}

		att := CreateAttestation(req.Subject, req.Predicate, "ai-model-bom", req.Evidence)
		if err := SignAttestation(&att, privKey); err != nil {
			log.Printf("sign attestation: %v", err)
			jsonError(w, "attestation signing failed", http.StatusInternalServerError)
			return
		}

		writeJSON(w, http.StatusOK, att)
	})

	mux.HandleFunc("/v1/metrics", func(w http.ResponseWriter, r *http.Request) {
		metricHTTPReqs.Add(1)
		if r.Method != http.MethodGet {
			jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		release, ok := beginProtectedRequest(w, r, token, limiter, concurrencySlots)
		if !ok {
			return
		}
		defer release()
		writeJSON(w, http.StatusOK, map[string]int64{
			"generated_total":     metricGenerated.Load(),
			"verified_total":      metricVerified.Load(),
			"diffs_total":         metricDiffs.Load(),
			"http_requests_total": metricHTTPReqs.Load(),
		})
	})

	return mux
}

func cmdKeygen(args []string) {
	fs := flag.NewFlagSet("keygen", flag.ExitOnError)
	prefix := fs.String("out", "ai-model-bom", "Output filename prefix")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ai-model-bom keygen [flags]\n\nGenerate an Ed25519 keypair for BOM signing.\n\nFlags:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		log.Fatalf("keygen failed: %v", err)
	}

	privB64 := base64.StdEncoding.EncodeToString(priv)
	pubB64 := base64.StdEncoding.EncodeToString(pub)

	privatePath := *prefix + ".key"
	publicPath := *prefix + ".pub"
	if err := writeNewFile(privatePath, []byte(privB64+"\n"), 0o600); err != nil {
		log.Fatalf("write private key: %v", err)
	}
	if err := writeNewFile(publicPath, []byte(pubB64+"\n"), 0o644); err != nil {
		if removeErr := os.Remove(privatePath); removeErr != nil {
			log.Printf("remove incomplete private key output: %v", removeErr)
		}
		log.Fatalf("write public key: %v", err)
	}

	fmt.Printf("Keys written: %s.key (private), %s.pub (public)\n", *prefix, *prefix)
}

// ---------- helpers ----------

func checkToken(r *http.Request, expected string) bool {
	if expected == "" {
		return false
	}
	auth := r.Header.Get("Authorization")
	scheme, provided, ok := strings.Cut(auth, " ")
	if !ok || scheme != "Bearer" || provided == "" || strings.ContainsAny(provided, " \t\r\n") {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(provided), []byte(expected)) == 1
}

func loadBOMFile(path string) (*ModelBOM, error) {
	return loadBOMFileWithPolicy(path, false)
}

func loadBOMFileWithPolicy(path string, allowUnsigned bool) (*ModelBOM, error) {
	data, err := readRegularFile(path, maxBOMBytes)
	if err != nil {
		return nil, err
	}

	var envelope map[string]json.RawMessage
	if err := decodeStrictJSON(data, &envelope); err != nil {
		return nil, fmt.Errorf("parse BOM: %w", err)
	}
	if _, signed := envelope["bom"]; signed {
		var sb SignedBOM
		if err := decodeStrictJSON(data, &sb); err != nil {
			return nil, fmt.Errorf("parse signed BOM: %w", err)
		}
		pubKey, err := loadVerifyKey()
		if err != nil {
			return nil, err
		}
		if pubKey == nil {
			return nil, errors.New("signed BOM input requires VERIFY_KEY or VERIFY_KEY_PATH")
		}
		if valid, reason := VerifySignedBOM(sb, pubKey); !valid {
			return nil, fmt.Errorf("signed BOM verification failed: %s", reason)
		}
		if hasBOMValidationErrors(sb.BOM) {
			return nil, errors.New("signed BOM failed structural validation")
		}
		return sb.BOM, nil
	}
	if !allowUnsigned {
		return nil, errors.New("unsigned BOM input requires --allow-unsigned")
	}

	var bom ModelBOM
	if err := decodeStrictJSON(data, &bom); err != nil {
		return nil, fmt.Errorf("parse BOM: %w", err)
	}
	if hasBOMValidationErrors(&bom) {
		return nil, errors.New("BOM failed structural validation")
	}
	return &bom, nil
}

func requiresSignedTrustInputs(cfg *ServiceConfig) bool {
	return cfg == nil || cfg.RequireSignedTrustInputs == nil || *cfg.RequireSignedTrustInputs
}

func decodeTrustedBOM(data []byte, verifyKey ed25519.PublicKey, requireSigned bool) (*ModelBOM, bool, error) {
	var envelope map[string]json.RawMessage
	if err := decodeStrictJSON(data, &envelope); err != nil {
		return nil, false, err
	}
	if _, signed := envelope["bom"]; signed {
		var sb SignedBOM
		if err := decodeStrictJSON(data, &sb); err != nil {
			return nil, false, err
		}
		if len(verifyKey) != ed25519.PublicKeySize {
			return nil, false, errors.New("signed BOM verification key is unavailable")
		}
		if valid, reason := VerifySignedBOM(sb, verifyKey); !valid {
			return nil, false, fmt.Errorf("signed BOM verification failed: %s", reason)
		}
		return sb.BOM, true, nil
	}
	if requireSigned {
		return nil, false, errors.New("signed BOM input is required")
	}
	var bom ModelBOM
	if err := decodeStrictJSON(data, &bom); err != nil {
		return nil, false, err
	}
	if hasBOMValidationErrors(&bom) {
		return nil, false, errors.New("BOM failed structural validation")
	}
	return &bom, false, nil
}

func isLoopbackBind(bindAddr string) bool {
	host, _, err := net.SplitHostPort(bindAddr)
	if err != nil {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// isPathAllowed checks if a path is under one of the allowed root directories.
// An empty allowlist denies every path.
func isPathAllowed(path string, allowedPaths []string) bool {
	_, allowed := resolveAllowedPath(path, allowedPaths)
	return allowed
}

func resolveAllowedPath(path string, allowedPaths []string) (string, bool) {
	roots, err := openAllowedRoots(allowedPaths)
	if err != nil {
		return "", false
	}
	defer closeAllowedRoots(roots)
	_, _, resolved, ok := resolveAllowedTarget(path, roots)
	return resolved, ok
}

func openAllowedRoots(paths []string) ([]*pinnedAllowedRoot, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	roots := make([]*pinnedAllowedRoot, 0, len(paths))
	seen := make(map[string]struct{}, len(paths))
	for _, configured := range paths {
		absolute, err := filepath.Abs(configured)
		if err != nil {
			closeAllowedRoots(roots)
			return nil, err
		}
		canonical, err := filepath.EvalSymlinks(absolute)
		if err != nil {
			closeAllowedRoots(roots)
			return nil, err
		}
		if _, duplicate := seen[canonical]; duplicate {
			continue
		}
		expected, err := os.Lstat(canonical)
		if err != nil || !expected.IsDir() || expected.Mode()&os.ModeSymlink != 0 {
			closeAllowedRoots(roots)
			return nil, fmt.Errorf("allowlist root %q is not a real directory", configured)
		}
		root, err := os.OpenRoot(canonical)
		if err != nil {
			closeAllowedRoots(roots)
			return nil, err
		}
		opened, err := root.Lstat(".")
		if err != nil || !opened.IsDir() || !os.SameFile(expected, opened) {
			_ = root.Close()
			closeAllowedRoots(roots)
			return nil, fmt.Errorf("allowlist root %q changed while opening", configured)
		}
		roots = append(roots, &pinnedAllowedRoot{canonicalPath: canonical, root: root})
		seen[canonical] = struct{}{}
	}
	return roots, nil
}

func closeAllowedRoots(roots []*pinnedAllowedRoot) {
	for _, allowed := range roots {
		if allowed != nil && allowed.root != nil {
			_ = allowed.root.Close()
		}
	}
}

func resolveAllowedTarget(path string, roots []*pinnedAllowedRoot) (*pinnedAllowedRoot, string, string, bool) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, "", "", false
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return nil, "", "", false
	}
	for _, allowed := range roots {
		if allowed == nil || allowed.root == nil {
			continue
		}
		if resolved != allowed.canonicalPath && !strings.HasPrefix(resolved, allowed.canonicalPath+string(filepath.Separator)) {
			continue
		}
		relative, err := filepath.Rel(allowed.canonicalPath, resolved)
		if err != nil || filepath.IsAbs(relative) || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			continue
		}
		if relative == "." {
			relative = "."
		}
		if _, err := allowed.root.Lstat(relative); err != nil {
			continue
		}
		return allowed, relative, resolved, true
	}
	return nil, "", "", false
}

// resolvePrivacyProfile returns a PrivacyProfile from a flag value.
func resolvePrivacyProfile(value string) (PrivacyProfile, error) {
	switch strings.ToLower(value) {
	case "none":
		return PrivacyProfile{}, nil
	case "default":
		return DefaultPrivacyProfile(), nil
	default:
		data, err := readRegularFile(value, maxConfigBytes)
		if err != nil {
			return PrivacyProfile{}, err
		}
		var profile PrivacyProfile
		if err := decodeStrictYAML(data, &profile); err != nil {
			return PrivacyProfile{}, err
		}
		return profile, nil
	}
}

func requireToken(w http.ResponseWriter, r *http.Request, expected string) bool {
	if expected == "" {
		jsonError(w, "service authentication is not configured", http.StatusServiceUnavailable)
		return false
	}
	if !checkToken(r, expected) {
		jsonError(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

type requestRateLimiter struct {
	mu          sync.Mutex
	limit       int
	count       int
	windowStart time.Time
}

func (l *requestRateLimiter) allow() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	if now.Sub(l.windowStart) >= time.Minute {
		l.windowStart = now
		l.count = 0
	}
	l.count++
	return l.count <= l.limit
}

func beginProtectedRequest(w http.ResponseWriter, r *http.Request, token string, limiter *requestRateLimiter, slots chan struct{}) (func(), bool) {
	if !requireToken(w, r, token) {
		return nil, false
	}
	if limiter == nil || !limiter.allow() {
		jsonError(w, "rate limit exceeded", http.StatusTooManyRequests)
		return nil, false
	}
	select {
	case slots <- struct{}{}:
		return func() { <-slots }, true
	default:
		jsonError(w, "too many concurrent requests", http.StatusTooManyRequests)
		return nil, false
	}
}

func decodeRequest(w http.ResponseWriter, r *http.Request, dst interface{}, maxBytes int64) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	return ensureJSONEOF(dec)
}

func jsonError(w http.ResponseWriter, message string, status int) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value interface{}) {
	data, err := json.Marshal(value)
	if err != nil {
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(append(data, '\n'))
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
		w.Header().Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (w *statusRecorder) WriteHeader(status int) {
	if w.status == 0 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusRecorder) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(data)
}

func auditHTTP(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		recorder := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(recorder, r)
		status := recorder.status
		if status == 0 {
			status = http.StatusOK
		}
		event := struct {
			Timestamp  string `json:"timestamp"`
			Method     string `json:"method"`
			Path       string `json:"path"`
			Status     int    `json:"status"`
			DurationMS int64  `json:"duration_ms"`
		}{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Method:    r.Method, Path: r.URL.Path, Status: status,
			DurationMS: time.Since(started).Milliseconds(),
		}
		if data, err := json.Marshal(event); err == nil {
			log.Printf("audit=%s", data)
		}
	})
}

func hasBOMValidationErrors(bom *ModelBOM) bool {
	for _, issue := range ValidateBOM(bom) {
		if issue.Severity == "error" {
			return true
		}
	}
	return false
}
