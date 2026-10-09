# Security policy

## Scope

This repository contains **local-only** redaction tooling. It does not ship credentials or call third-party paid APIs.

## Reporting vulnerabilities

Please report security issues privately to the repository maintainers (do not open public issues for exploit details).

## Safe usage

1. Run redaction on trusted machines; output may still leak context if detectors miss faces/plates.
2. Never commit `.env`, API keys, or raw customer video to git.
3. When enabling `FACE_BLUR_ALLOW_NETWORK_MODELS`, download SCRFD/optional weights only from trusted HTTPS endpoints with pinned SHA-256 digests.
4. Review every output frame before publication or model training.

## Out of scope for this repo

Cloud inference credentials, internal admin URLs, and proprietary routing layers are intentionally omitted.
