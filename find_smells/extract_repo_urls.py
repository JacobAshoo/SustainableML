"""
Extract unique repository URLs from each smell CSV into separate files.
"""

import csv
import os

SMELL_CSVS = {
    "unbounded_graph_expansion.csv": "unbounded_graph_expansion_repos.txt",
    "graph_constant_bottleneck.csv": "graph_constant_bottleneck_repos.txt",
    "gpu_released_memory_failure.csv": "gpu_released_memory_failure_repos.txt",
    "shape_mismatch_leak.csv": "shape_mismatch_leak_repos.txt",
}

for csv_file, out_file in SMELL_CSVS.items():
    if not os.path.exists(csv_file):
        print(f"[!] {csv_file} not found, skipping.")
        continue

    urls = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("repo_url", "").strip()
            if url:
                urls.add(url)

    sorted_urls = sorted(urls)
    with open(out_file, "w", encoding="utf-8") as f:
        for url in sorted_urls:
            f.write(url + "\n")

    print(f"{csv_file} → {out_file}  ({len(sorted_urls)} unique repos)")
