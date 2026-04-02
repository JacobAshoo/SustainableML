"""
Smell 1 Detector: Unbounded Graph Expansion
Detects patterns where TensorFlow graph nodes are created inside loops
without clearing/resetting the graph, leading to continuous graph growth.

Detection patterns:
- tf.* operations inside for/while loops without tf.reset_default_graph() or tf.Graph()
- Session.run() inside loops with graph-building ops in the same loop
- Missing tf.compat.v1.reset_default_graph() in loop bodies
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

OUTPUT_FILE = "unbounded_graph_expansion.csv"
TARGET_FILES = ["train.py", "main.py", "model.py", "run.py", "trainer.py",
                "training.py", "experiment.py", "pipeline.py"]
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


# ─── Smell Detection Logic ───────────────────────────────────────────────────
TF_OP_PATTERNS = [
    r"tf\.Variable\(",
    r"tf\.constant\(",
    r"tf\.placeholder\(",
    r"tf\.matmul\(",
    r"tf\.add\(",
    r"tf\.multiply\(",
    r"tf\.nn\.\w+\(",
    r"tf\.layers\.\w+\(",
    r"tf\.keras\.layers\.\w+\(",
    r"tf\.compat\.v1\.\w+\(",
    r"tf\.get_variable\(",
    r"tf\.train\.\w+Optimizer\(",
    r"tf\.reduce_\w+\(",
    r"tf\.concat\(",
    r"tf\.reshape\(",
    r"tf\.expand_dims\(",
    r"tf\.squeeze\(",
]

GRAPH_RESET_PATTERNS = [
    r"tf\.reset_default_graph\(\)",
    r"tf\.compat\.v1\.reset_default_graph\(\)",
    r"tf\.Graph\(\)",
    r"graph\.as_default\(\)",
    r"with\s+tf\.Graph\(\)",
]


def get_context_snippet(lines, line_idx, context=2):
    start = max(0, line_idx - context)
    end = min(len(lines), line_idx + context + 1)
    snippet_lines = []
    for j in range(start, end):
        marker = ">>>" if j == line_idx else "   "
        snippet_lines.append(f"{marker} {j + 1}: {lines[j]}")
    return "\n".join(snippet_lines)


def detect_unbounded_graph_expansion(content, file_path):
    findings = []
    lines = content.split("\n")

    in_loop = False
    loop_start = -1
    loop_indent = -1
    loop_has_tf_ops = []
    loop_has_reset = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        loop_match = re.match(r"^(\s*)(for |while )", line)
        if loop_match:
            if in_loop and loop_has_tf_ops and not loop_has_reset:
                for op_line, op_snippet in loop_has_tf_ops:
                    snippet = get_context_snippet(lines, op_line, context=2)
                    findings.append((
                        op_line + 1,
                        snippet,
                        f"TF graph operation inside loop (started line {loop_start + 1}) "
                        f"without tf.reset_default_graph() or tf.Graph() context manager. "
                        f"This causes unbounded graph expansion."
                    ))

            in_loop = True
            loop_start = i
            loop_indent = len(loop_match.group(1))
            loop_has_tf_ops = []
            loop_has_reset = False
            continue

        if in_loop:
            current_indent = len(line) - len(line.lstrip()) if stripped else loop_indent + 1
            if stripped and current_indent <= loop_indent:
                if loop_has_tf_ops and not loop_has_reset:
                    for op_line, op_snippet in loop_has_tf_ops:
                        snippet = get_context_snippet(lines, op_line, context=2)
                        findings.append((
                            op_line + 1,
                            snippet,
                            f"TF graph operation inside loop (started line {loop_start + 1}) "
                            f"without tf.reset_default_graph() or tf.Graph() context manager. "
                            f"This causes unbounded graph expansion."
                        ))
                in_loop = False
                loop_has_tf_ops = []
                loop_has_reset = False

        if in_loop:
            for pattern in GRAPH_RESET_PATTERNS:
                if re.search(pattern, stripped):
                    loop_has_reset = True
                    break

            for pattern in TF_OP_PATTERNS:
                if re.search(pattern, stripped):
                    loop_has_tf_ops.append((i, stripped))
                    break

    if in_loop and loop_has_tf_ops and not loop_has_reset:
        for op_line, op_snippet in loop_has_tf_ops:
            snippet = get_context_snippet(lines, op_line, context=2)
            findings.append((
                op_line + 1,
                snippet,
                f"TF graph operation inside loop (started line {loop_start + 1}) "
                f"without tf.reset_default_graph() or tf.Graph() context manager. "
                f"This causes unbounded graph expansion."
            ))

    return findings


# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Smell 1 Detector: Unbounded Graph Expansion")
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

                    if "import tensorflow" not in content and "from tensorflow" not in content:
                        continue

                    findings = detect_unbounded_graph_expansion(content, file_path)

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
                        print(f"    [SMELL FOUND] Line {line_num}: Unbounded graph expansion (total: {total_findings})")

        page += 1
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Done. {total_findings} smells detected across {repos_checked} repositories.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
