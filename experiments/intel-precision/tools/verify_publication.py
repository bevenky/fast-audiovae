"""Verify saved Intel evidence and source identity without numerical execution."""
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "experiments/intel-precision"
EVIDENCE = REPO / "benchmarks/intel-precision"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    publication = read(EVIDENCE / "publication.json")
    for name, record in publication["files"].items():
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO) and path.is_file(), "Unsafe publication path")
        require(digest(path) == record["published_sha256"], "Published file changed: " + name)
        require(path.stat().st_size == record["bytes"], "Published size changed: " + name)

    original = read(BUNDLE / "pins/original-release-manifest.json")
    preserved = 0
    for name, record in original["files"].items():
        if name.split("/")[0] in {"native", "fused", "support"} or name.startswith("tools/"):
            require(digest(BUNDLE / name) == record["sha256"], "Accepted source changed: " + name)
            preserved += 1

    archive = BUNDLE / "archive/iteration3"
    source_map = read(archive / "source-manifest.json")["source_files"]
    archive_hashes = set()
    for name, record in source_map.items():
        path = (archive / name).resolve()
        require(path.is_relative_to(archive) and path.is_file(), "Unsafe archive path")
        value = digest(path)
        require(value == record["published_sha256"], "Archived file changed: " + name)
        if path.suffix != ".json":
            require(value == record["source_sha256"], "Archived source bytes changed: " + name)
        archive_hashes.add(value)

    build_sources = 0
    omitted_metadata = 0
    for path in (EVIDENCE / "iteration3").glob("build-*/build.json"):
        for name, value in read(path)["input_sha256"].items():
            if ("/core/" in name or "/pipeline/" in name) and Path(name).suffix in {".h", ".c", ".cpp", ".py"}:
                if Path(name).name.startswith("._"):
                    # Historical AppleDouble metadata was listed, never compiled.
                    omitted_metadata += 1
                    continue
                require(value in archive_hashes, "Measured source missing from archive: " + name)
                build_sources += 1

    summary = read(EVIDENCE / "summary.json")
    campaign = read(EVIDENCE / "campaign-results.json")
    require(campaign["status"] == "complete" and not campaign["failures"], "Incomplete campaign")
    require(len(campaign["checks"]) == 340 and len(campaign["measurements"]) == 250,
            "Campaign counts differ")
    require(summary["checks"]["failed"] == 0, "Reported failed validation")
    require(len(summary["timing_protocol"]["validation_uids"]) == 60, "Validation corpus differs")
    require(summary["source_sha256"]["timing"] == publication["files"]["benchmarks/intel-precision/campaign-results.json"]["source_sha256"], "Timing provenance differs")
    require(summary["source_sha256"]["quality"] == publication["files"]["benchmarks/intel-precision/quality/comparison.json"]["source_sha256"], "Quality provenance differs")
    for name, expected in {"audio_stock": 0.657688, "fast_fp32": 0.248212,
                           "int8_large": 0.164109, "mimi": 0.170021}.items():
        require(round(summary["timing"][name]["decoder_rtf"], 6) == expected, "README RTF differs")

    rejected = read(EVIDENCE / "iteration3/summary.json")
    require(len(rejected["screens"]) == 7 and not rejected["promoted"], "Rejected screen status differs")
    for screen in rejected["screens"]:
        name = "benchmarks/intel-precision/iteration3/" + screen["variant"] + "/results.json"
        require(publication["files"][name]["source_sha256"] == screen["report_sha256"], "Screen provenance differs")
        result = read(REPO / name)
        require(result["status"] == "complete" and not result["failures"], "Incomplete screen")
        require(len(result["checks"]) == 67 and len(result["measurements"]) == 30, "Screen counts differ")
        require(not screen["summary"]["accepted"] and not screen["summary"]["meets_10pct_aggregate"], "Unexpected promotion")

    parsed = 0
    for path in BUNDLE.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path.relative_to(REPO)))
        parsed += 1
    print(json.dumps({"status": "verified", "published_records": len(publication["files"]),
                      "accepted_source_files_unchanged": preserved,
                      "archived_source_files": len(source_map), "measured_build_source_records": build_sources,
                      "omitted_appledouble_metadata": omitted_metadata, "python_files_parsed": parsed,
                      "validation_rows": 340, "timed_calls": 250, "rejected_screens": 7,
                      "native_execution": False, "model_execution": False, "new_timings": False}))


if __name__ == "__main__":
    main()
