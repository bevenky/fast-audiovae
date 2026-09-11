"""Pinned Freesound mirror metadata only; four workers, hard 1 GiB read cap."""
import concurrent.futures
import json
from pathlib import Path
import re
import threading
import time
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

REPO = "MoamenElSayed/freesound-commercial-50k"
REV = "ac5aed8cf1aeb97b26a375f24794d42e15f97163"
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
LIMIT = 1_000_000_000
counts = {"read_bytes": 0, "completed_shards": 0, "rows": 0}
lock = threading.Lock()
columns = ["title", "description", "tags", "username", "freesound_id", "license", "attribution_required", "commercial_use"]

class CountedFile:
    def __init__(self, inner):
        self.inner = inner
        self.regions = []
    def read(self, n=-1):
        if n < 0: raise ValueError("Unbounded metadata reads forbidden")
        position = self.inner.tell()
        for start, data in self.regions:
            if start <= position and position + n <= start + len(data):
                self.inner.seek(position + n)
                return data[position-start:position-start+n]
        with lock:
            if counts["read_bytes"] + n > LIMIT: raise ValueError("Metadata read limit reached")
            counts["read_bytes"] += n
        return self.inner.read(n)
    def preload(self, start, end):
        previous = self.inner.tell()
        self.inner.seek(start)
        data = self.read(end-start)
        self.regions.append((start, data))
        self.inner.seek(previous)
    def seek(self, *args): return self.inner.seek(*args)
    def tell(self): return self.inner.tell()
    def readable(self): return True
    def writable(self): return False
    def seekable(self): return True
    def close(self): self.inner.close()
    @property
    def closed(self): return self.inner.closed


def scan(index):
    name = f"data/train-{index:05d}-of-00167.parquet"
    result = OUT / f"metadata-{index:05d}.json"
    if result.exists(): return json.loads(result.read_text())
    fs = HfFileSystem()
    remote = f"datasets/{REPO}@{REV}/{name}"
    with fs.open(remote, "rb", block_size=65536, cache_type="none") as stream:
        counted = CountedFile(stream)
        file = pq.ParquetFile(counted, pre_buffer=False)
        found = []
        offset = 0
        groups = []
        for group in range(file.num_row_groups):
            info = file.metadata.row_group(group)
            groups.append({"rows": info.num_rows, "audio_compressed_bytes": info.column(0).total_compressed_size})
            ranges, audio_ranges = [], []
            for column_index in range(info.num_columns):
                column = info.column(column_index)
                start = column.dictionary_page_offset or column.data_page_offset
                pair = (start, start + column.total_compressed_size)
                (ranges if column.path_in_schema.split('.')[0] in columns else audio_ranges).append(pair)
            start, end = min(x[0] for x in ranges), max(x[1] for x in ranges)
            if any(a < end and b > start for a, b in audio_ranges):
                raise ValueError("Metadata prefetch would include an audio column")
            counted.preload(start, end)
            for local, row in enumerate(file.read_row_group(group, columns=columns, use_threads=False).to_pylist()):
                text = " ".join((row["title"] or "", row["description"] or "", " ".join(row["tags"] or [])))
                if re.search(r"whistl|whiss?l|silbid", text, re.I):
                    found.append(dict(row, shard=name, row=offset+local, row_group=group, row_within_group=local))
            offset += info.num_rows
        payload = {"shard": name, "rows": offset, "groups": groups, "candidates": found}
        result.write_text(json.dumps(payload, indent=2))
        with lock:
            counts["completed_shards"] += 1
            counts["rows"] += offset
            print(json.dumps(counts), flush=True)
        return payload


def main():
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        all_shards = list(pool.map(scan, range(167)))
    candidates = [row for shard in all_shards for row in shard["candidates"]]
    (OUT / "hf-whistle-candidates.json").write_text(json.dumps(candidates, indent=2))
    summary = dict(counts, revision=REV, repo=REPO, matched_rows=len(candidates), elapsed_seconds=time.time()-start,
                   policy="metadata columns only; no audio column reads; maximum four workers")
    (OUT / "metadata-scan.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

if __name__ == "__main__": main()
