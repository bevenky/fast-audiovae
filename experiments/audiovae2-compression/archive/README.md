# Archive preparation

`prepare_archive.py` inventories candidate text and copies only when passed `--copy`. It does not edit source files, run experiments, or modify Git. Each invocation requires a new manifest path. Copy mode checks every destination before writing and rejects any existing file, including identical files.

From the workspace root:

```sh
python3 outputs/archive-preparation/prepare_archive.py --manifest outputs/archive-preparation/final-selection.json
python3 outputs/archive-preparation/prepare_archive.py --copy --manifest outputs/archive-preparation/copied-selection.json
```

Review the selected records and the `sensitive_or_embedded_payload`, `oversize`, `bulk_catalog_manifest_only` and `aggregate_projection` categories before copying. No secret values are printed. Credential signature matches can include public web-page tokens and require classification rather than assuming a private credential leak.

After copying, place this directory's `ARCHIVE.md` at the repository root and retain the copy manifest with it. The root document's links are relative to the repository root. Keep the archive script with the manifest so the projection rules remain reproducible.

Scope covers all `outputs/convnext*`, `outputs/audiovae2-compression*`, external `work/convnext*` helper trees except environments/caches, and the two AudioVAE2 compression helper trees. It excludes unrelated CPU runtime histories already archived elsewhere. Large binary deployment bundles are never copied; only eligible text members are examined and mirrored.

The method HTML is copied byte-for-byte. Its relative link to `work/fast-audiovae/docs/cpu-kernel-results.md` is satisfied by a matching archived documentation copy. External live TensorBoard links remain historical references. Local linked aggregate reports may be labelled projections at the same relative URL.
