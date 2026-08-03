# Security audit — 2026-08-02

## Scope

This audit covered model discovery and hashing, BOM generation and validation,
signatures and attestations, deployment-readiness decisions, strict input
handling, HTTP authentication and authorization, configuration, persistence,
containerization, dependencies, and release automation.

## Remediated findings

- Made configuration, JSON, YAML, metadata, evidence, privacy-profile, and BOM
  parsing bounded and strict, including rejection of unknown fields and
  trailing or multiple documents.
- Added regular-file, symlink, size, and file-identity checks around sensitive
  reads and streaming model hashes; supported model errors now fail generation
  instead of silently producing an incomplete inventory.
- Made file outputs restrictive, atomic, durable, and resistant to symlink or
  accidental-overwrite attacks. Key generation also cleans up partial output.
- Enforced valid Ed25519 key sizes, safe signature error handling, mandatory
  signing for attestations, and verification of signed BOMs before trust-bearing
  diff, attestation, or readiness operations. BOM signatures now authenticate
  the signer, timestamp, and signature-format version as well as the BOM hash.
- Made daemon authentication fail closed with an exact, constant-time compared,
  minimum-length token and owner-only secret-file support.
- Made the generation allowlist mandatory, verified each configured root, and
  constrained requested model paths to those roots.
- Added bounded HTTP decoding, generic client errors, security headers,
  conservative server timeouts, graceful shutdown, per-process rate limiting,
  and structured request audit events.
- Strengthened readiness checks so failed evidence and non-approved promotion
  status cannot pass a production policy. Evidence required for readiness or a
  signed attestation must identify its type, source, timestamp, and result, and
  readiness requires at least one valid passing record.
- Canonicalized allowlisted request paths before scanning, rejected symlinked
  generic inputs, and bounded directory traversal to 200,000 entries.
- Replaced pathname walking with descriptor-backed, batched `os.Root`
  traversal. Directory symlinks and special files now fail the inventory, and
  file identity is checked before and after hashing.
- Pinned allowlist roots when the daemon starts and made generation consume a
  root-relative target through that descriptor. This closes the authorization-
  to-open race where replacing an ancestor could previously redirect a
  canonicalized request outside the approved tree.
- Trust-bearing CLI and daemon diff/readiness operations require verified
  signed BOMs by default; unsigned operation is an explicit structural-only
  opt-in. Attestation signatures are domain-separated.
- Split ordinary service access from private-key use. Signed generation and
  attestation now require a distinct signing-authority token, fail closed when
  it is absent, and the daemon rejects reuse of the normal service token. This
  prevents evaluation clients from minting trusted claims or self-asserted
  promotion metadata.
- Added comprehensive nested-field and dependency bounds, dangling/self/
  duplicate dependency rejection, a concurrent-request cap, and explicit
  remote-bind opt-in.
- Resolved every medium-or-higher `gosec` finding; variable paths are confined
  to descriptor-backed roots or narrowly justified operator-selected readers
  with regular-file, identity, and size checks.
- Hardened the pinned, non-root container image and added SHA-pinned CI,
  CodeQL, dependency update, SBOM, provenance, attestation, and signed release
  automation.
- Gated every release publisher on a read-only check that requires a
  v-prefixed strict SemVer annotated tag resolving exactly to the workflow
  commit, with that commit contained in `origin/main`. Write, package, OIDC,
  and attestation permissions remain confined to the publishing jobs.

## Verification performed

- `go test -race -count=1 ./...`
- `go vet ./...`
- `gosec -severity medium -confidence medium ./...`: no findings
- `govulncheck ./...`
- Actionlint on all workflows
- Release dependency/permission graph checks, strict SemVer positive/negative
  fixtures, and synthetic annotated/lightweight/off-main Git ancestry cases
- Gitleaks across Git history and the working tree
- Trivy source, secret, configuration, and final-image scans
- Read-only, capability-dropped, no-new-privileges container smoke test

The exact final results are recorded in the parent SecAI-Hub audit handoff for
this work batch.

## Residual and operational risks

- Rate limiting is per process; use an aggregate gateway control for multiple
  replicas.
- Standard-output audit records need a protected, durable external log sink.
- Secret values supplied directly through environment variables may be exposed
  to privileged operators; prefer the `_PATH` options and a secrets manager.
- Model storage and BOM output need encrypted volumes and backups when they
  contain sensitive or customer data.
- Filesystem identity checks cannot establish trust in an actively hostile
  shared filesystem; production model mounts should be immutable and read-only.
- URL fields are validated but remote content is not fetched or attested by this
  service. A separate trusted ingestion step should establish provenance.
