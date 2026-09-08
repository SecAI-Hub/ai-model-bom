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

Go 1.26.8 or newer is required. CI and container builds use Go 1.27.1.

```bash
# Build
go build -o ai-model-bom .

# Generate a BOM from a model file
ai-model-bom generate ./models/mistral-7b.gguf --meta metadata.yaml --out bom.json

# Sign the BOM (the private-key file must not be group/world accessible)
export SIGNING_KEY_PATH="$PWD/ai-model-bom.key"
ai-model-bom generate ./models/mistral-7b.gguf --out signed-bom.json

# Verify a signed BOM
export VERIFY_KEY_PATH="$PWD/ai-model-bom.pub"
ai-model-bom verify signed-bom.json

# Compare two BOMs
ai-model-bom diff old-signed-bom.json new-signed-bom.json

# Evaluate deployment readiness
ai-model-bom evaluate signed-bom.json

# Run as HTTP daemon
ai-model-bom serve --config policies/default-policy.yaml
```

## Key generation

```bash
ai-model-bom keygen --out ai-model-bom
# Produces: ai-model-bom.key (private), ai-model-bom.pub (public)
```

## HTTP API

All endpoints except `/health` require an exact bearer token. Verification,
diff, evaluation, and metrics use `SERVICE_TOKEN`. When a private signing key is
configured, `/v1/generate` and `/v1/attest` instead require the distinct
`SIGNING_SERVICE_TOKEN`; they fail closed if that signing-authority token is not
configured. The daemon rejects equal service and signing tokens. Tokens must
contain at least 32 non-whitespace bytes, and `daemon.allowed_paths` must contain
at least one existing directory.

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
| `SERVICE_TOKEN` | Bearer token for protected endpoints (minimum 32 bytes) |
| `SERVICE_TOKEN_PATH` | Owner-only regular file containing the bearer token |
| `SIGNING_SERVICE_TOKEN` | Distinct bearer token authorized to use the private signing key |
| `SIGNING_SERVICE_TOKEN_PATH` | Owner-only regular file containing the signing-authority token |
| `SIGNING_KEY` | Base64-encoded Ed25519 private key |
| `SIGNING_KEY_PATH` | Owner-only regular file containing the private key |
| `VERIFY_KEY` | Base64-encoded Ed25519 public key |
| `VERIFY_KEY_PATH` | Regular file containing the public key |

Set either a direct value or its `_PATH` variant, never both. File-based
secrets avoid exposing credentials through process environments. The server
uses a separate verification key; it does not derive trust from a caller's
payload.

Possession of the signing-service token is equivalent to signing authority: it
can mint signed BOMs from allowlisted model paths and signed attestations from
validated evidence. Keep it separate from ordinary evaluation clients, rotate
it independently, and omit it entirely on verification-only deployments.

Configuration and data files are parsed with size limits, symlink rejection,
unknown-field rejection, and single-document enforcement. An empty
`allowed_paths` list denies all daemon generation requests. `diff`, `attest`,
and `evaluate` CLI commands require verified signed BOM inputs by default;
`--allow-unsigned` is an explicit structural-only escape hatch. Daemon
`/v1/diff` and `/v1/evaluate` likewise require signed inputs by default and the
service refuses to start without a verification key when
`require_signed_trust_inputs` is enabled. Signed BOM signatures cover the
content hash, signer, timestamp, and signature format version. Attestation
signatures use a separate domain-separated payload.

The daemon pins every allowlisted directory as an `os.Root` when it starts and
performs later generation through that descriptor and a confined relative path.
Replacing an ancestor after authorization therefore cannot redirect hashing
outside the approved tree. Directory generation also uses bounded batched
enumeration and pre/post identity checks; symlinks and special entries are
rejected instead of being silently omitted from the inventory.

Evidence used for readiness or signed attestations must have a recognized type,
source, RFC 3339 timestamp, and result. A readiness policy requiring `evidence`
requires at least one structurally valid passing item; failed evidence always
blocks readiness.

The daemon applies separate evaluation and signing authorities, per-process
request and concurrency limits, emits
structured request audit events to standard output, and sets conservative HTTP
timeouts and security headers. Aggregate rate limiting and durable log
collection should be enforced at the deployment boundary.

## Docker

```bash
docker build -t ai-model-bom .
chmod 600 service-token
docker run --rm --read-only --cap-drop=ALL \
  --user "$(id -u):$(id -g)" \
  --security-opt=no-new-privileges --pids-limit=64 \
  -p 127.0.0.1:8515:8515 \
  -v "$PWD/service-token:/run/secrets/service-token:ro" \
  -v "$PWD/ai-model-bom.pub:/run/secrets/verify-key:ro" \
  -v "$PWD/models:/models:ro" \
  -e SERVICE_TOKEN_PATH=/run/secrets/service-token \
  -e VERIFY_KEY_PATH=/run/secrets/verify-key \
  ai-model-bom
```

The bundled container policy permits `/models` and listens on all container
interfaces; the example publishes it only on host loopback. Mount additional
policy/key files and a distinct owner-only signing-service token read-only when
HTTP signing is needed. Overriding the image user
with the host file owner's UID/GID lets the process read owner-only bind-mounted
secrets without making them group/world-readable. If the default UID 65534 is
used instead, secret files must be securely projected with that ownership.

## Testing

```bash
go test -race -count=1 ./...
go vet ./...
```

## Security

See [SECURITY.md](SECURITY.md) for the security design and vulnerability
reporting process, and [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the latest
audit and remaining operational risks.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
