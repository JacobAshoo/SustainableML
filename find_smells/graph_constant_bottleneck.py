"""
Smell 2 Detector: Graph-Constant Bottleneck
Detects patterns where large constants or datasets are embedded directly
into the TensorFlow computational graph instead of using tf.data, feeds, or variables.

Detection patterns:
- tf.constant() with large arrays, lists, or numpy data
- Loading entire datasets into tf.constant()
- Embedding numpy arrays directly as graph constants
- Using tf.constant() with file reads (np.load, pd.read_csv, etc.)
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

OUTPUT_FILE = "graph_constant_bottleneck.csv"
TARGET_FILES = ["train.py", "main.py", "model.py", "run.py", "trainer.py",
                "training.py", "experiment.py", "pipeline.py", "data.py",
                "dataset.py", "utils.py", "preprocess.py"]
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

DATA_LOAD_PATTERNS = [
    r"np\.load\(",
    r"np\.loadtxt\(",
    r"np\.genfromtxt\(",
    r"pd\.read_csv\(",
    r"pd\.read_excel\(",
    r"np\.array\(",
    r"np\.zeros\(",
    r"np\.ones\(",
    r"np\.random\.\w+\(",
    r"pickle\.load\(",
    r"json\.load\(",
    r"scipy\.io\.loadmat\(",
    r"h5py\.File\(",
    r"open\(.+\.read\(\)",
]

LARGE_CONSTANT_PATTERNS = [
    (r"tf\.constant\(\s*(\w+)", "tf.constant() used with variable '{var}' — "
     "if this holds a large dataset/array, it embeds it directly into the graph, "
     "causing memory bloat. Use tf.placeholder + feed_dict or tf.data.Dataset instead."),

    (r"tf\.constant\(\s*np\.\w+", "tf.constant() wrapping a NumPy operation directly embeds "
     "the resulting array into the computation graph. Large arrays should be fed "
     "via tf.data.Dataset or tf.placeholder."),

    (r"tf\.constant\(\s*\[.{100,}\]", "tf.constant() with a large inline list embeds the data "
     "directly into the graph definition, bloating memory. Use tf.Variable or "
     "tf.data for large data."),

    (r"tf\.convert_to_tensor\(\s*(\w+)", "tf.convert_to_tensor() used with variable '{var}' — "
     "converts data into a graph constant. If the data is large, this causes "
     "graph-constant bottleneck. Use tf.data pipeline instead."),
]


def detect_graph_constant_bottleneck(content, file_path):
    findings = []
    lines = content.split("\n")

    # Track variables that hold loaded data
    data_vars = set()

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("#"):
            continue

        # Track data loading into variables
        for pattern in DATA_LOAD_PATTERNS:
            if re.search(pattern, stripped):
                assign_match = re.match(r"(\w+)\s*=", stripped)
                if assign_match:
                    data_vars.add(assign_match.group(1))

        # Check for large constant patterns
        for pattern, explanation_template in LARGE_CONSTANT_PATTERNS:
            match = re.search(pattern, stripped)
            if match:
                var_name = match.group(1) if match.lastindex else ""
                explanation = explanation_template.replace("{var}", var_name)

                if var_name in data_vars:
                    explanation += " (Variable was loaded from file/data source.)"

                snippet = get_context_snippet(lines, i, context=3)
                findings.append((i + 1, snippet, explanation))
                break

    # Detect tf.constant near data loading blocks
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "tf.constant" in stripped or "tf.convert_to_tensor" in stripped:
            for j in range(max(0, i - 5), i):
                prev_line = lines[j].strip()
                for pattern in DATA_LOAD_PATTERNS:
                    if re.search(pattern, prev_line):
                        if not any(f[0] == i + 1 for f in findings):
                            snippet = get_context_snippet(lines, i, context=3)
                            findings.append((
                                i + 1,
                                snippet,
                                f"tf.constant/convert_to_tensor near data loading "
                                f"(line {j + 1}). Loaded data may be embedded directly "
                                f"into the graph, causing graph-constant bottleneck."
                            ))
                        break

    return findings


# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Smell 2 Detector: Graph-Constant Bottleneck")
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

                    findings = detect_graph_constant_bottleneck(content, file_path)

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
                        print(f"    [SMELL FOUND] Line {line_num}: Graph-constant bottleneck (total: {total_findings})")

        page += 1
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Done. {total_findings} smells detected across {repos_checked} repositories.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
