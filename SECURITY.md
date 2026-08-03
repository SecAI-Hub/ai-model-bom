# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | Yes                |
| < 0.2   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in ai-model-bom, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@secai-hub.dev**

Include:
- A description of the vulnerability
- Steps to reproduce
- Impact assessment
- Any suggested fixes

We will acknowledge receipt within 48 hours and provide an initial assessment within 5 business days.

## Security Design

ai-model-bom follows defense-in-depth principles:

- **Authentication and signing authority**: All non-health HTTP endpoints require an exact bearer token of at least 32 bytes. Private-key operations use a distinct signing-service token; the daemon rejects an equal ordinary token and disables signing routes when the privileged token is absent. Tokens may be loaded from owner-only files.
- **Path allowlisting**: The daemon refuses to start without an existing directory in `allowed_paths`, pins each root descriptor at startup, and performs `/v1/generate` reads relative to that `os.Root`; later ancestor replacement cannot redirect hashing outside the approved tree.
- **Symlink and special-file rejection**: Model/config/input readers reject symlinks, devices, FIFOs, sockets, and files that change during use; directory inventories are bounded and batched.
- **File size limits**: Supported model files over the configured hard limit fail generation instead of disappearing from the resulting BOM.
- **Strict bounded decoding**: HTTP request bodies and JSON/YAML files have hard limits; unknown fields, trailing values, and multiple YAML documents are rejected.
- **Server controls**: `http.Server` enforces header, read, write, and idle timeouts, security headers, graceful shutdown, and per-process request/concurrency limits. Non-loopback binding requires `ALLOW_REMOTE_BIND=true`.
- **Constant-time token comparison**: `crypto/subtle.ConstantTimeCompare` prevents timing attacks
- **Ed25519 signing**: BOMs and domain-separated attestations are signed with validated Ed25519 keys; the BOM signature covers signer and timestamp metadata. Trust-bearing operations require verified signed inputs by default, with explicit unsigned structural-only opt-in.
- **Evidence quality gates**: Signed attestations and readiness evidence require a recognized type, source, timestamp, and result; failed evidence and non-approved promotion states fail closed.
- **Durable output**: File output uses restrictive creation modes, symlink refusal, atomic replacement, and file/directory synchronization.
- **Privacy redaction**: Evidence can be stripped of emails, paths, hostnames, and usernames before export
- **Input validation**: URLs, SPDX licenses, timestamps, and enums are validated before use
- **Non-root execution**: The container runs as UID 65534 (nobody)

## Threat Model

See the parent project's [threat model](https://github.com/SecAI-Hub/SecAI_OS/blob/main/docs/threat-model.md) for the full system-level analysis.

Key threats specific to ai-model-bom:
- **BOM tampering**: Mitigated by Ed25519 signatures and SHA-256 content hashes
- **Path traversal**: Mitigated by allowlisted root directories and symlink rejection
- **PII leakage in evidence**: Mitigated by configurable privacy redaction profiles
- **Unauthorized BOM generation**: Mitigated by service token authentication

## Operational Limits

- Store signing keys and service tokens in a secrets manager or owner-only
  mounted files. Treat the signing-service token as key-use authority and keep
  it distinct from evaluation clients. Environment variables may be visible to
  privileged local operators and some orchestration systems.
- Request throttling is local to each process. Apply an aggregate limit at the
  reverse proxy or service mesh for multi-replica deployments.
- The HTTP service is plaintext. Keep the default loopback bind or terminate
  authenticated TLS and enforce network policy before remote exposure.
- Request audit records are written to standard output. Route them to a
  protected, access-controlled, integrity-retaining log sink.
- Model bytes and generated BOMs are not encrypted by this application. Use
  encrypted storage and transport appropriate to the data classification.
- Filesystem identity checks reduce time-of-check/time-of-use risk but cannot
  make a hostile, mutable shared filesystem trustworthy. Prefer immutable,
  read-only model mounts.
- Directory traversal is bounded to 200,000 entries and model files to 10 GiB;
  apply tighter job-level CPU, memory, I/O, and wall-clock limits where possible.
