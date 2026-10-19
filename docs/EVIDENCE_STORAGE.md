# Exact evidence storage

Scientific analysis uses the original artifact bytes and their declared SHA-256
hashes. Some aggregate JSON collections repeat retained requests and raw responses
and can become too large for a normal GitHub push. New large aggregates may be
stored in Git as a lossless gzip file plus an `.archive.json` sidecar. This does
not change a request, annotation, criterion, denominator or analysis input.

Use the repository utility on a completed, immutable artifact:

```bash
.venv/bin/python scripts/preserve_large_evidence_v1.py --archive audits/path/first_collection.json
```

It creates `first_collection.json.gz` and `first_collection.json.archive.json`,
verifies full decompression against the original size and SHA-256, and leaves the
original file intact. Compression has a fixed timestamp and no embedded filename.
The utility never changes Git configuration or stages files.

Commit the archive, sidecar and first execution evidence. If the original large
aggregate is not already tracked, exclude only that generated path after verifying
its archive; keep analysis dependencies bound to its original byte hash. Retain
individual first-response files, manifests, receipts, logs and all failures under
their existing policies. Do not rewrite already published history or replace old
raw artifacts to reduce repository size.

On a fresh checkout, restore the exact analysis input before invoking its normal
hash-bound collector or assessment:

```bash
.venv/bin/python scripts/preserve_large_evidence_v1.py --restore audits/path/first_collection.json.archive.json
```

Restore verifies both compressed and uncompressed size/hash, installs the original
atomically and refuses to overwrite different existing work. An identical existing
file is verified and left in place. Archive paths remain within repository
`audits/`. A successful round trip preserves bytes; it does not validate the
scientific contents. The first nine fictional utility checks pass, including
tampering and existing-file protections.
