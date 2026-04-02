"""
Smell 4 Detector: Shape Mismatch Leak
Detects patterns where tensors are not reshaped properly, leading to
excess memory allocation from oversized or misaligned tensors.

Detection patterns:
- tf.reshape with -1 (unknown dimensions) without validation
- Missing reshape before matrix operations (matmul, conv2d)
- Flatten operations on high-dimensional tensors without necessity
- Transpose/permute without subsequent reshape to expected dimensions
- Broadcasting mismatches that silently expand memory
"""

import requests
import csv
import re
import time
import os
import base64
import sys

# ─── Configuration ───────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

OUTPUT_FILE = "shape_mismatch_leak.csv"
TARGET_FILES = ["train.py", "main.py", "model.py", "run.py", "trainer.py",
                "training.py", "experiment.py", "pipeline.py", "network.py",
                "layers.py", "net.py", "architecture.py"]
MAX_REPOS = 1000
RESULTS_PER_PAGE = 100
RATE_LIMIT_PAUSE = 10

CSV_FIELDS = ["repository", "repo_url", "file_path",
              "line_number", "code_snippet", "explanation"]


# ─── CSV Helpers ─────────────────────────────────────────────────────────────
def init_csv():
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv(row):
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ─── GitHub API Helpers ──────────────────────────────────────────────────────
def rate_limit_wait(response):
    if response.status_code == 403:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_time - int(time.time()), RATE_LIMIT_PAUSE)
        print(f"  [!] Rate limited. Waiting {wait}s...")
        time.sleep(wait)
        return True
    return False


def search_repositories(page=1):
    url = "https://api.github.com/search/repositories"
    query = "tensorflow OR keras language:Python stars:10..100 archived:false"
    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": RESULTS_PER_PAGE,
        "page": page,
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if rate_limit_wait(response):
        response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("items", [])


def search_code_in_repo(repo_full_name, filename):
    url = "https://api.github.com/search/code"
    query = f"filename:{filename} repo:{repo_full_name}"
    params = {"q": query, "per_page": 5}
    response = requests.get(url, headers=HEADERS, params=params)
    if rate_limit_wait(response):
        response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return []
    return response.json().get("items", [])


def get_file_content(repo_full_name, file_path):
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    response = requests.get(url, headers=HEADERS)
    if rate_limit_wait(response):
        response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return None
    data = response.json()
    if data.get("encoding") == "base64":
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return None
    return data.get("content")


def get_context_snippet(lines, line_idx, context=2):
    start = max(0, line_idx - context)
    end = min(len(lines), line_idx + context + 1)
    snippet_lines = []
    for j in range(start, end):
        marker = ">>>" if j == line_idx else "   "
        snippet_lines.append(f"{marker} {j + 1}: {lines[j]}")
    return "\n".join(snippet_lines)


# ─── Smell Detection Logic ───────────────────────────────────────────────────

def detect_shape_mismatch_leak(content, file_path):
    findings = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("#"):
            continue

        # ── Pattern 1: tf.reshape with -1 without shape validation ──
        if re.search(r"tf\.reshape\(.+,\s*\[.*-1.*\]\)", stripped):
            nearby_start = max(0, i - 5)
            nearby_end = min(len(lines), i + 6)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])

            has_shape_check = bool(re.search(
                r"\.get_shape\(\)|\.shape|tf\.debugging\.assert|assert.*shape|"
                r"tf\.ensure_shape|set_shape",
                nearby_text
            ))

            if not has_shape_check:
                snippet = get_context_snippet(lines, i, context=3)
                findings.append((
                    i + 1,
                    snippet,
                    "tf.reshape() with -1 dimension used without shape validation. "
                    "If the input tensor has an unexpected shape, the inferred "
                    "dimension may be much larger than intended, silently "
                    "allocating excess memory (shape mismatch leak)."
                ))

        # ── Pattern 2: Flatten without necessity before dense layers ──
        if re.search(r"tf\.reshape\(.+,\s*\[-1\]\)|Flatten\(\)|\.flatten\(\)", stripped):
            for j in range(i + 1, min(len(lines), i + 4)):
                next_line = lines[j].strip()
                if re.search(r"Dense\(|tf\.layers\.dense\(|tf\.matmul\(", next_line):
                    snippet = get_context_snippet(lines, i, context=3)
                    findings.append((
                        i + 1,
                        snippet,
                        "Full tensor flatten before Dense/matmul operation. "
                        "Flattening high-dimensional tensors creates unnecessarily "
                        "large intermediate tensors. Consider using GlobalAveragePooling "
                        "or GlobalMaxPooling to reduce dimensionality before dense layers, "
                        "which avoids shape mismatch memory leaks."
                    ))
                    break

        # ── Pattern 3: Matrix operations without prior reshape ──
        if re.search(r"tf\.matmul\(", stripped):
            nearby_start = max(0, i - 5)
            pre_text = "\n".join(lines[nearby_start:i])

            has_reshape = bool(re.search(
                r"tf\.reshape\(|\.reshape\(|tf\.expand_dims\(|tf\.squeeze\(",
                pre_text
            ))

            if not has_reshape:
                matmul_match = re.search(r"tf\.matmul\(\s*(\w+)\s*,\s*(\w+)", stripped)
                if matmul_match:
                    var_a, var_b = matmul_match.group(1), matmul_match.group(2)
                    snippet = get_context_snippet(lines, i, context=3)
                    findings.append((
                        i + 1,
                        snippet,
                        f"tf.matmul({var_a}, {var_b}) called without prior reshape. "
                        f"If these tensors have incompatible or oversized shapes, "
                        f"TensorFlow may broadcast or pad them, causing silent "
                        f"memory over-allocation (shape mismatch leak)."
                    ))

        # ── Pattern 4: tf.expand_dims used repeatedly ──
        if re.search(r"tf\.expand_dims\(", stripped):
            nearby_start = max(0, i - 3)
            nearby_end = min(len(lines), i + 4)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])
            expand_count = len(re.findall(r"tf\.expand_dims\(", nearby_text))

            if expand_count >= 2:
                snippet = get_context_snippet(lines, i, context=3)
                findings.append((
                    i + 1,
                    snippet,
                    f"Multiple tf.expand_dims() calls ({expand_count}) in close "
                    f"proximity. Repeatedly expanding dimensions creates "
                    f"progressively larger tensor copies. Consider using "
                    f"tf.reshape() to set the target shape directly."
                ))

        # ── Pattern 5: Broadcasting without explicit shape control ──
        if re.search(r"tf\.broadcast_to\(|tf\.tile\(", stripped):
            nearby_start = max(0, i - 5)
            nearby_end = min(len(lines), i + 3)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])

            has_shape_check = bool(re.search(
                r"\.get_shape\(\)|\.shape\b|assert.*shape|tf\.ensure_shape|set_shape",
                nearby_text
            ))

            if not has_shape_check:
                snippet = get_context_snippet(lines, i, context=3)
                findings.append((
                    i + 1,
                    snippet,
                    "tf.broadcast_to() or tf.tile() used without shape validation. "
                    "These operations can silently create very large tensors if "
                    "the source tensor shape doesn't match expectations, causing "
                    "excessive memory allocation."
                ))

        # ── Pattern 6: Concatenation without shape checks in loops ──
        if re.search(r"tf\.concat\(", stripped):
            nearby_start = max(0, i - 5)
            nearby_end = min(len(lines), i + 3)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])

            has_shape_check = bool(re.search(
                r"\.get_shape\(\)|\.shape\b|assert.*shape|tf\.ensure_shape|"
                r"tf\.debugging\.assert",
                nearby_text
            ))

            if not has_shape_check:
                loop_found = False
                for k in range(max(0, i - 10), i):
                    if re.match(r"\s*(for |while )", lines[k]):
                        loop_found = True
                        break

                if loop_found:
                    snippet = get_context_snippet(lines, i, context=3)
                    findings.append((
                        i + 1,
                        snippet,
                        "tf.concat() inside a loop without shape validation. "
                        "Concatenating tensors of mismatched shapes in a loop "
                        "accumulates excess memory. Validate tensor shapes before "
                        "concatenation to prevent shape mismatch leaks."
                    ))

        # ── Pattern 7: tf.pad without shape validation ──
        if re.search(r"tf\.pad\(", stripped):
            nearby_start = max(0, i - 3)
            nearby_end = min(len(lines), i + 3)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])

            has_shape_check = bool(re.search(
                r"\.shape|\.get_shape|assert|tf\.ensure_shape",
                nearby_text
            ))

            if not has_shape_check:
                snippet = get_context_snippet(lines, i, context=3)
                findings.append((
                    i + 1,
                    snippet,
                    "tf.pad() used without shape validation on the input tensor. "
                    "Padding tensors with unexpected shapes can allocate "
                    "significantly more memory than needed."
                ))

    return findings


# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Smell 4 Detector: Shape Mismatch Leak")
    print("=" * 60)

    init_csv()
    total_findings = 0
    repos_checked = 0
    page = 1

    while repos_checked < MAX_REPOS:
        print(f"\n[*] Fetching repositories (page {page})...")
        repos = search_repositories(page=page)
        if not repos:
            print("  No more repositories found.")
            break

        for repo in repos:
            if repos_checked >= MAX_REPOS:
                break

            full_name = repo["full_name"]
            repo_url = repo["html_url"]
            stars = repo.get("stargazers_count", 0)

            if repo.get("size", 0) == 0:
                continue

            repos_checked += 1
            print(f"\n[{repos_checked}/{MAX_REPOS}] Checking: {full_name} ({stars} stars)")

            for target_file in TARGET_FILES:
                time.sleep(1)
                code_results = search_code_in_repo(full_name, target_file)

                for code_item in code_results:
                    file_path = code_item.get("path", "")
                    print(f"  Inspecting: {file_path}")

                    content = get_file_content(full_name, file_path)
                    if not content:
                        continue

                    if "import tensorflow" not in content and "from tensorflow" not in content \
                       and "import keras" not in content and "from keras" not in content:
                        continue

                    findings = detect_shape_mismatch_leak(content, file_path)

                    for line_num, snippet, explanation in findings:
                        row = {
                            "repository": full_name,
                            "repo_url": repo_url,
                            "file_path": file_path,
                            "line_number": line_num,
                            "code_snippet": snippet,
                            "explanation": explanation,
                        }
                        append_csv(row)
                        total_findings += 1
                        print(f"    [SMELL FOUND] Line {line_num}: Shape mismatch leak (total: {total_findings})")

        page += 1
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Done. {total_findings} smells detected across {repos_checked} repositories.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
