"""
Smell 3 Detector: GPU Released Memory Failure
Detects patterns where GPU memory is not properly released after use,
leading to memory leaks and OOM errors.

Detection patterns:
- Missing tf.keras.backend.clear_session()
- Sessions not closed (no sess.close() or context manager)
- No gpu_options.allow_growth or memory fraction config
- Missing tf.compat.v1.reset_default_graph() between training runs
- Reliance on gc.collect() for GPU memory without explicit GPU cleanup
- Creating multiple sessions/models without releasing GPU memory
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

OUTPUT_FILE = "gpu_released_memory_failure.csv"
TARGET_FILES = ["train.py", "main.py", "model.py", "run.py", "trainer.py",
                "training.py", "experiment.py", "pipeline.py", "utils.py"]
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

GPU_CLEANUP_PATTERNS = [
    r"tf\.keras\.backend\.clear_session\(\)",
    r"K\.clear_session\(\)",
    r"keras\.backend\.clear_session\(\)",
    r"tf\.compat\.v1\.reset_default_graph\(\)",
    r"sess\.close\(\)",
    r"session\.close\(\)",
    r"with\s+tf\.Session\(",
    r"with\s+tf\.compat\.v1\.Session\(",
    r"tf\.config\.experimental\.set_memory_growth",
    r"gpu_options\.allow_growth\s*=\s*True",
    r"per_process_gpu_memory_fraction",
    r"del\s+model",
    r"del\s+sess",
]


def detect_gpu_memory_failure(content, file_path):
    findings = []
    lines = content.split("\n")
    full_text = content

    # ── Check 1: Session created without close or context manager ──
    session_lines = []
    has_session_close = False
    uses_context_manager = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if re.search(r"tf\.Session\(|tf\.compat\.v1\.Session\(|tf\.InteractiveSession\(", stripped):
            if re.search(r"^\s*with\s+", stripped):
                uses_context_manager = True
            else:
                session_lines.append(i)

        if re.search(r"\.close\(\)", stripped):
            has_session_close = True

    if session_lines and not has_session_close and not uses_context_manager:
        for sl in session_lines:
            snippet = get_context_snippet(lines, sl, context=2)
            findings.append((
                sl + 1,
                snippet,
                "TF Session created without using a context manager (with statement) "
                "and no explicit .close() call found. GPU memory allocated by the session "
                "will not be released, causing GPU memory leaks."
            ))

    # ── Check 2: Multiple model builds without clear_session ──
    model_create_lines = []
    has_clear_session = bool(re.search(
        r"clear_session\(\)|reset_default_graph\(\)", full_text
    ))

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if re.search(
            r"tf\.keras\.Sequential\(|tf\.keras\.Model\(|keras\.models\.\w+\(|"
            r"Sequential\(\[|Model\(inputs",
            stripped
        ):
            model_create_lines.append(i)

    if len(model_create_lines) > 1 and not has_clear_session:
        for ml in model_create_lines:
            snippet = get_context_snippet(lines, ml, context=2)
            findings.append((
                ml + 1,
                snippet,
                f"Multiple model instantiations found ({len(model_create_lines)} total) "
                f"without tf.keras.backend.clear_session() or "
                f"tf.compat.v1.reset_default_graph(). Each new model accumulates GPU "
                f"memory from prior models, leading to GPU memory leaks."
            ))

    # ── Check 3: gc.collect() used for GPU cleanup without proper GPU release ──
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if re.search(r"gc\.collect\(\)", stripped):
            nearby_start = max(0, i - 5)
            nearby_end = min(len(lines), i + 5)
            nearby_text = "\n".join(lines[nearby_start:nearby_end])

            has_gpu_cleanup = False
            for pattern in GPU_CLEANUP_PATTERNS:
                if re.search(pattern, nearby_text):
                    has_gpu_cleanup = True
                    break

            if not has_gpu_cleanup:
                snippet = get_context_snippet(lines, i, context=2)
                findings.append((
                    i + 1,
                    snippet,
                    "gc.collect() used without explicit GPU memory release (e.g., "
                    "tf.keras.backend.clear_session(), del model, or "
                    "tf.config.experimental.set_memory_growth). Python's garbage "
                    "collector primarily handles CPU memory and is unreliable for "
                    "GPU memory deallocation."
                ))

    # ── Check 4: Training loops without GPU memory management ──
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(for|while)\s+", stripped):
            loop_indent = len(line) - len(line.lstrip())
            j = i + 1
            has_training = False
            has_cleanup = False

            while j < len(lines):
                inner = lines[j]
                inner_stripped = inner.strip()
                if inner_stripped:
                    inner_indent = len(inner) - len(inner.lstrip())
                    if inner_indent <= loop_indent:
                        break

                if re.search(r"\.fit\(|\.train\(|\.train_on_batch\(|sess\.run\(", inner_stripped):
                    has_training = True

                for pattern in GPU_CLEANUP_PATTERNS:
                    if re.search(pattern, inner_stripped):
                        has_cleanup = True
                        break

                j += 1

            if has_training and not has_cleanup:
                snippet = get_context_snippet(lines, i, context=2)
                findings.append((
                    i + 1,
                    snippet,
                    "Training/session.run() calls inside a loop without GPU memory "
                    "cleanup (clear_session, del model, reset_default_graph). "
                    "Repeated training runs accumulate GPU memory, leading to "
                    "OOM errors."
                ))

    # ── Check 5: No GPU memory config at all in files using GPU ──
    uses_gpu_keywords = bool(re.search(
        r"\.fit\(|\.train\(|sess\.run\(|model\.compile\(|gpu|cuda|GPU",
        full_text
    ))
    has_any_gpu_config = bool(re.search(
        r"set_memory_growth|allow_growth|per_process_gpu_memory_fraction|"
        r"visible_devices|CUDA_VISIBLE_DEVICES|clear_session",
        full_text
    ))

    if uses_gpu_keywords and not has_any_gpu_config:
        import_line = 0
        for i, line in enumerate(lines):
            if "import tensorflow" in line or "from tensorflow" in line:
                import_line = i
                break
        snippet = get_context_snippet(lines, import_line, context=2)
        findings.append((
            import_line + 1,
            snippet,
            "File uses TensorFlow with GPU operations but has no GPU memory "
            "configuration (allow_growth, set_memory_growth, "
            "per_process_gpu_memory_fraction). By default TF allocates all "
            "GPU memory, which may not be released properly."
        ))

    return findings


# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Smell 3 Detector: GPU Released Memory Failure")
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

                    findings = detect_gpu_memory_failure(content, file_path)

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
                        print(f"    [SMELL FOUND] Line {line_num}: GPU memory failure (total: {total_findings})")

        page += 1
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Done. {total_findings} smells detected across {repos_checked} repositories.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
