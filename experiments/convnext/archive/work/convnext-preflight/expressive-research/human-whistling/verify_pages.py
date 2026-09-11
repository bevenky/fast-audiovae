"""Fetch public source-page metadata only, never preview or original audio."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
MAX_PAGE_BYTES = 2_000_000


class Evidence(HTMLParser):
    def __init__(self, sound_id):
        super().__init__()
        self.sound_id = str(sound_id)
        self.value = {}

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        if tag == "meta" and a.get("name") == "twitter:description":
            self.value["description"] = a.get("content", "")
        if a.get("data-sound-id") == self.sound_id:
            self.value.update(title=a.get("data-title"), duration_seconds=float(a["data-duration"]),
                              sample_rate_hz=float(a["data-samplerate"]))
        if tag == "a" and a.get("title") == "Go to the full license text":
            self.value["license_url"] = a.get("href")


def verify(row):
    sound_id = row["freesound_id"]
    url = f"https://freesound.org/s/{sound_id}/"
    result = dict(row, source_url=url)
    try:
        page_path = ROOT / f"freesound-{sound_id}.html"
        with urllib.request.urlopen(url, timeout=35) as response:
            payload = response.read(MAX_PAGE_BYTES + 1)
            if len(payload) > MAX_PAGE_BYTES:
                raise ValueError("Source page exceeds metadata limit")
            result["resolved_source_url"] = response.url
        page_path.write_bytes(payload)
        parser = Evidence(sound_id)
        parser.feed(payload.decode("utf-8"))
        evidence = parser.value
        license_url = evidence.get("license_url", "").replace("http://", "https://")
        allowed = {"https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
                   "https://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
                   "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0"}
        if license_url not in allowed:
            raise ValueError(f"Source page license is not in the permitted set: {license_url}")
        if "duration_seconds" not in evidence:
            raise ValueError("Source page does not identify this audio ID")
        result.update(verified_license=allowed[license_url], source_evidence=evidence,
                      source_page_sha256=sha256(payload).hexdigest(), source_page_path=str(page_path),
                      verified_at_utc=datetime.now(timezone.utc).isoformat(), state="metadata_verified")
    except Exception as error:
        result.update(state="verification_failed", error=f"{type(error).__name__}: {error}")
    print(sound_id, result["state"], flush=True)
    return result


if __name__ == "__main__":
    rows = json.loads((ROOT / "provisional.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(verify, rows))
    (ROOT / "source-evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps({"checked": len(evidence), "verified": sum(r["state"] == "metadata_verified" for r in evidence)}))
