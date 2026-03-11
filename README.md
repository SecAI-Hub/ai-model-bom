# ai-model-bom

Security-grade AI/ML Bill of Materials generator, verifier, and attestation companion for [SecAI_OS](https://github.com/SecAI-Hub/SecAI_OS).

## Features

- **BOM generation** -- CycloneDX-aligned ML-BOM from model files or directories with SHA-256 hashes
- **Ed25519 signing & verification** -- tamper-proof BOMs and attestations
- **BOM diff** -- compare two BOMs for trust-relevant drift (hash, license, lineage changes)
- **Evidence attestation** -- attach and verify test/audit evidence on model components
- **Deployment readiness evaluation** -- check required fields before model promotion
- **Privacy redaction** -- strip emails, paths, hostnames, and usernames from evidence
- **HTTP daemon** -- authenticated REST API for CI/CD integration
- **Input validation** -- URLs, SPDX licenses, timestamps, and enum fields validated at ingestion

## Quick start

```bash
# Build
go build -o ai-model-bom .

# Generate a BOM from a model file
ai-model-bom generate ./models/mistral-7b.gguf --meta metadata.yaml --out bom.json

# Sign the BOM
export SIGNING_KEY=$(cat ai-model-bom.key)
ai-model-bom generate ./models/mistral-7b.gguf --out signed-bom.json

# Verify a signed BOM
export VERIFY_KEY=$(cat ai-model-bom.pub)
ai-model-bom verify signed-bom.json

# Compare two BOMs
ai-model-bom diff old-bom.json new-bom.json

# Evaluate deployment readiness
ai-model-bom evaluate bom.json

# Run as HTTP daemon
ai-model-bom serve --config policies/default-policy.yaml
```

## Key generation

```bash
ai-model-bom keygen --out ai-model-bom
# Produces: ai-model-bom.key (private), ai-model-bom.pub (public)
```

## HTTP API

All endpoints except `/health` require `Authorization: Bearer <SERVICE_TOKEN>`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/generate` | Generate BOM from a model path |
| POST | `/v1/verify` | Verify a signed BOM |
| POST | `/v1/diff` | Compare two BOMs |
| POST | `/v1/evaluate` | Check deployment readiness |
| POST | `/v1/attest` | Create a signed attestation |
| GET | `/v1/metrics` | Prometheus-style counters |

## Configuration

See [policies/default-policy.yaml](policies/default-policy.yaml) for all options.

| Env var | Description |
|---------|-------------|
| `AIMBOM_CONFIG` | Path to config YAML (default: `policies/default-policy.yaml`) |
| `SERVICE_TOKEN` | Bearer token for protected endpoints |
| `SIGNING_KEY` | Base64-encoded Ed25519 private key |
| `VERIFY_KEY` | Base64-encoded Ed25519 public key |

## Docker

```bash
docker build -t ai-model-bom .
docker run -p 8515:8515 -e SERVICE_TOKEN=mytoken ai-model-bom serve
```

## Testing

```bash
go test -v -race ./...
```

## Security

See [SECURITY.md](SECURITY.md) for the security design and vulnerability reporting process.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
