"""
Combined ML Code Smell Detector — Code Search Edition
Uses GitHub Code Search API to find specific code patterns directly,
then fetches files via raw.githubusercontent.com (free CDN) to verify.

Outputs 4 CSVs, one per smell, with results saved on every find.

Usage:
    export GITHUB_TOKEN="ghp_..."
    python all_smells_detector.py
"""

import requests
import csv
import re
import time
import os

# ─── Configuration ───────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

RATE_LIMIT_PAUSE = 10
MAX_PAGES_PER_QUERY = 10  # GitHub caps at 1000 results (10 pages × 100)
PER_PAGE = 100
MIN_STARS = 10
MAX_STARS = 500

OUTPUT_FILES = {
    "smell1": "unbounded_graph_expansion.csv",
    "smell2": "graph_constant_bottleneck.csv",
    "smell3": "gpu_released_memory_failure.csv",
    "smell4": "shape_mismatch_leak.csv",
}

CSV_FIELDS = ["repository", "repo_url", "file_path",
              "line_number", "code_snippet", "explanation"]

# Track files already processed to avoid duplicates across queries
processed_files = set()


# ─── CSV Helpers ─────────────────────────────────────────────────────────────
def init_all_csvs():
    for path in OUTPUT_FILES.values():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv(smell_key, row):
    with open(OUTPUT_FILES[smell_key], "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ─── GitHub API / CDN Helpers ────────────────────────────────────────────────
def api_get(url, params=None):
    """GitHub API request with rate-limit retry."""
    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), RATE_LIMIT_PAUSE)
            print(f"  [!] Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 422:
            print(f"  [!] Query validation failed (422). Skipping.")
            return resp
        return resp


def code_search(query, page=1):
    """
    Search code via GitHub Code Search API.
    NOTE: Code search only supports: language, repo, org, user, path,
    filename, extension. It does NOT support stars, size, etc.
    Stars are filtered post-search from repo metadata.
    """
    resp = api_get("https://api.github.com/search/code", {
        "q": query,
        "per_page": PER_PAGE,
        "page": page,
    })
    if resp.status_code != 200:
        print(f"  [!] Search returned {resp.status_code}")
        return [], 0
    data = resp.json()
    return data.get("items", []), data.get("total_count", 0)


def get_repo_stars(full_name):
    """Fetch star count for a repo (uses regular API rate limit)."""
    resp = api_get(f"https://api.github.com/repos/{full_name}")
    if resp.status_code == 200:
        return resp.json().get("stargazers_count", 0)
    return -1


def get_raw_content(full_name, file_path, branch="master"):
    """Fetch file via raw.githubusercontent.com — CDN, no API rate limit."""
    url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{file_path}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404 and branch == "master":
            return get_raw_content(full_name, file_path, "main")
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


# ─── Utility ─────────────────────────────────────────────────────────────────
def get_snippet(lines, idx, context=2):
    start = max(0, idx - context)
    end = min(len(lines), idx + context + 1)
    parts = []
    for j in range(start, end):
        marker = ">>>" if j == idx else "   "
        parts.append(f"{marker} {j + 1}: {lines[j]}")
    return "\n".join(parts)


def file_key(full_name, path):
    return f"{full_name}:{path}"


# ─── Star Cache ──────────────────────────────────────────────────────────────
# Cache star counts so we only fetch once per repo
star_cache = {}


def check_stars(full_name):
    """Return True if repo has stars in [MIN_STARS, MAX_STARS], with caching."""
    if full_name in star_cache:
        return star_cache[full_name]

    stars = get_repo_stars(full_name)
    result = MIN_STARS <= stars <= MAX_STARS
    star_cache[full_name] = result

    if not result:
        if stars == -1:
            print(f"    Skipping {full_name} (could not fetch stars)")
        else:
            print(f"    Skipping {full_name} ({stars} stars, need {MIN_STARS}-{MAX_STARS})")
    return result


# ─── Search + Verify Framework ──────────────────────────────────────────────
def search_and_verify(queries, verify_fn, smell_key):
    """
    For each query: paginate through code search results,
    check star count, fetch content via CDN, run verify_fn.
    """
    total = 0

    for query in queries:
        # language:python is valid for code search
        full_query = f"{query} language:python"
        print(f"\n  Query: {full_query}")

        for page in range(1, MAX_PAGES_PER_QUERY + 1):
            # Code search: 30 req/min — pause between calls
            time.sleep(2)

            items, count = code_search(full_query, page)
            if not items:
                break

            if page == 1:
                print(f"  Found {count} results")

            for item in items:
                repo = item.get("repository", {})
                full_name = repo.get("full_name", "")
                fpath = item.get("path", "")
                fk = file_key(full_name, fpath)

                # Skip already processed
                if fk in processed_files:
                    continue
                processed_files.add(fk)

                # Check stars (cached per repo)
                if not check_stars(full_name):
                    continue

                # Fetch content via CDN (free)
                branch = repo.get("default_branch", "master")
                content = get_raw_content(full_name, fpath, branch)
                if not content:
                    continue

                # Must import TF/Keras
                if ("import tensorflow" not in content
                        and "from tensorflow" not in content
                        and "import keras" not in content
                        and "from keras" not in content):
                    continue

                lines = content.split("\n")
                findings = verify_fn(content, lines)

                if findings:
                    print(f"  Scanning: {full_name}/{fpath}")

                for line_num, snippet, explanation in findings:
                    repo_url = repo.get("html_url", f"https://github.com/{full_name}")
                    append_csv(smell_key, {
                        "repository": full_name,
                        "repo_url": repo_url,
                        "file_path": fpath,
                        "line_number": line_num,
                        "code_snippet": snippet,
                        "explanation": explanation,
                    })
                    total += 1
                    print(f"    [FOUND] {full_name} — {fpath}:{line_num} (total: {total})")

            if len(items) < PER_PAGE:
                break

    return total


# ══════════════════════════════════════════════════════════════════════════════
# SMELL 1: Unbounded Graph Expansion
# Graph-building ops (tf.Variable, tf.placeholder, tf.get_variable, tf.layers,
# tf.nn) inside for/while loops with no graph reset.
# ══════════════════════════════════════════════════════════════════════════════
SMELL1_QUERIES = [
    # Each query requires BOTH terms to appear somewhere in the file
    "tf.Variable for in range",
    "tf.Variable while",
    "tf.placeholder for in range",
    "tf.placeholder while",
    "tf.get_variable for in range",
    "tf.get_variable while",
    "tf.layers for in range",
    "tf.nn for in range",
    "tf.keras.layers for in range",
]

GRAPH_OPS_RE = [
    r"tf\.Variable\(", r"tf\.constant\(", r"tf\.placeholder\(",
    r"tf\.get_variable\(", r"tf\.layers\.\w+\(", r"tf\.nn\.\w+\(",
    r"tf\.keras\.layers\.\w+\(", r"tf\.train\.\w+Optimizer\(",
]
GRAPH_RESETS_RE = [
    r"tf\.reset_default_graph\(", r"tf\.compat\.v1\.reset_default_graph\(",
    r"tf\.Graph\(\)", r"\.as_default\(\)",
]


def verify_smell1(content, lines):
    findings = []
    in_loop = False
    loop_start = -1
    loop_indent = -1
    loop_ops = []
    loop_reset = False

    def flush():
        nonlocal in_loop, loop_ops, loop_reset
        if loop_ops and not loop_reset:
            for ol in loop_ops:
                findings.append((
                    ol + 1, get_snippet(lines, ol),
                    f"TF graph-building op inside loop (line {loop_start + 1}) "
                    f"with no graph reset. New nodes added every iteration — "
                    f"unbounded graph expansion and memory growth."
                ))
        in_loop = False
        loop_ops = []
        loop_reset = False

    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        lm = re.match(r"^(\s*)(for |while )", line)
        if lm:
            if in_loop:
                flush()
            in_loop = True
            loop_start = i
            loop_indent = len(lm.group(1))
            loop_ops = []
            loop_reset = False
            continue

        if in_loop:
            ci = len(line) - len(line.lstrip())
            if ci <= loop_indent and s:
                flush()
                continue
            for p in GRAPH_RESETS_RE:
                if re.search(p, s):
                    loop_reset = True
                    break
            for p in GRAPH_OPS_RE:
                if re.search(p, s):
                    loop_ops.append(i)
                    break

    if in_loop:
        flush()
    return findings


# ══════════════════════════════════════════════════════════════════════════════
# SMELL 2: Graph-Constant Bottleneck
# File-loaded data passed to tf.constant / tf.convert_to_tensor
# ══════════════════════════════════════════════════════════════════════════════
SMELL2_QUERIES = [
    "tf.constant np.load",
    "tf.constant np.loadtxt",
    "tf.constant np.genfromtxt",
    "tf.constant pd.read_csv",
    "tf.constant pd.read_excel",
    "tf.constant pickle.load",
    "tf.constant cv2.imread",
    "tf.constant imageio.imread",
    "tf.constant h5py",
    "tf.convert_to_tensor np.load",
    "tf.convert_to_tensor pd.read_csv",
    "tf.convert_to_tensor pickle.load",
    "tf.convert_to_tensor cv2.imread",
]

FILE_LOADS = [
    "np.load", "np.loadtxt", "np.genfromtxt", "np.fromfile",
    "pd.read_csv", "pd.read_excel", "pd.read_parquet", "pd.read_hdf",
    "pickle.load", "joblib.load", "scipy.io.loadmat", "h5py.File",
    "cv2.imread", "imageio.imread",
]
FILE_LOAD_RE = re.compile(r"(" + "|".join(re.escape(c) for c in FILE_LOADS) + r")\s*\(")


def verify_smell2(content, lines):
    findings = []

    # Track variables from file loads
    data_vars = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            continue
        if FILE_LOAD_RE.search(s):
            m = re.match(r"(\w+)\s*=", s)
            if m:
                data_vars.add(m.group(1))

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            continue

        # Inline: tf.constant(np.load(...))
        if re.search(r"tf\.constant\(", s) and FILE_LOAD_RE.search(s):
            load = FILE_LOAD_RE.search(s)
            findings.append((
                i + 1, get_snippet(lines, i, 3),
                f"tf.constant() wraps {load.group(1)}(). Entire file contents "
                f"embedded into graph as a constant permanently. "
                f"Use tf.data.Dataset or tf.placeholder instead."
            ))
            continue

        if re.search(r"tf\.convert_to_tensor\(", s) and FILE_LOAD_RE.search(s):
            load = FILE_LOAD_RE.search(s)
            findings.append((
                i + 1, get_snippet(lines, i, 3),
                f"tf.convert_to_tensor() wraps {load.group(1)}(). "
                f"Loaded data becomes permanent graph constant. "
                f"Use tf.data.Dataset or tf.placeholder instead."
            ))
            continue

        # Variable: tf.constant(data) where data = np.load(...)
        cm = re.search(r"tf\.constant\(\s*(\w+)\s*[,)]", s)
        if cm and cm.group(1) in data_vars:
            v = cm.group(1)
            findings.append((
                i + 1, get_snippet(lines, i, 3),
                f"tf.constant({v}) where '{v}' holds file-loaded data. "
                f"Entire dataset embedded permanently in graph. "
                f"Use tf.data.Dataset or tf.placeholder + feed_dict."
            ))
            continue

        cm = re.search(r"tf\.convert_to_tensor\(\s*(\w+)\s*[,)]", s)
        if cm and cm.group(1) in data_vars:
            v = cm.group(1)
            findings.append((
                i + 1, get_snippet(lines, i, 3),
                f"tf.convert_to_tensor({v}) where '{v}' holds file-loaded data. "
                f"Data becomes permanent graph constant. "
                f"Use tf.data.Dataset or tf.placeholder."
            ))

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# SMELL 3: GPU Released Memory Failure
# Sessions without close/context manager, multiple models without clear_session,
# training inside loops without cleanup
# ══════════════════════════════════════════════════════════════════════════════
SMELL3_QUERIES = [
    # Session leak patterns
    "tf.Session tensorflow",
    "tf.InteractiveSession tensorflow",
    "tf.compat.v1.Session tensorflow",
    # Training in loops
    "model.fit for in range keras",
    "model.fit for epoch keras",
    "model.fit while keras",
    "sess.run for in range tensorflow",
    "train_on_batch for in range keras",
    # Multiple models
    "Sequential Sequential keras",
]

GPU_CLEANUP_RE = [
    r"tf\.keras\.backend\.clear_session\(", r"K\.clear_session\(",
    r"keras\.backend\.clear_session\(",
    r"tf\.compat\.v1\.reset_default_graph\(", r"tf\.reset_default_graph\(",
    r"\.close\(\)", r"del\s+model", r"del\s+sess",
    r"tf\.config\.experimental\.set_memory_growth",
]


def verify_smell3(content, lines):
    findings = []
    full = content

    # ── Check 1: Session without close or context manager ──
    has_close = bool(re.search(r"\.close\(\)", full))
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            continue
        if re.search(r"=\s*tf\.(Session|compat\.v1\.Session|InteractiveSession)\(", s):
            if not re.search(r"^\s*with\s+", s) and not has_close:
                findings.append((
                    i + 1, get_snippet(lines, i),
                    "TF Session assigned without .close() or context manager. "
                    "GPU memory is never released."
                ))

    # ── Check 2: Multiple model instantiations without clear_session ──
    has_clear = bool(re.search(r"clear_session\(|reset_default_graph\(", full))
    model_lines = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            continue
        if re.search(
            r"=\s*(tf\.keras\.Sequential|tf\.keras\.Model|keras\.models\.\w+|Sequential)\s*\(", s
        ) or re.search(r"=\s*Model\s*\(\s*inputs", s):
            model_lines.append(i)

    if len(model_lines) > 1 and not has_clear:
        for ml in model_lines:
            findings.append((
                ml + 1, get_snippet(lines, ml),
                f"{len(model_lines)} model instantiations with no clear_session() "
                f"or reset_default_graph(). Each model accumulates GPU memory "
                f"from prior models — never freed."
            ))

    # ── Check 3: Training inside loops without GPU cleanup ──
    for i, line in enumerate(lines):
        s = line.strip()
        if not re.match(r"(for|while)\s+", s):
            continue
        loop_indent = len(line) - len(line.lstrip())
        j = i + 1
        has_train = False
        has_cleanup = False

        while j < len(lines):
            inner = lines[j]
            si = inner.strip()
            if si:
                if len(inner) - len(inner.lstrip()) <= loop_indent:
                    break
            if re.search(r"\.fit\(|\.train_on_batch\(|sess\.run\(", si):
                has_train = True
            for p in GPU_CLEANUP_RE:
                if re.search(p, si):
                    has_cleanup = True
                    break
            j += 1

        if has_train and not has_cleanup:
            findings.append((
                i + 1, get_snippet(lines, i),
                "Training calls inside loop with no GPU memory cleanup "
                "(clear_session/del model/reset_default_graph/sess.close). "
                "Each iteration accumulates GPU memory → OOM."
            ))

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# SMELL 4: Shape Mismatch Leak
# Flatten before Dense, tf.concat accumulation in loops,
# 3+ consecutive expand_dims
# ══════════════════════════════════════════════════════════════════════════════
SMELL4_QUERIES = [
    "Flatten Dense keras",
    "Flatten Dense tensorflow",
    "flatten Dense keras",
    "tf.concat for in range tensorflow",
    "tf.concat while tensorflow",
    "tf.expand_dims tf.expand_dims tensorflow",
]


def verify_smell4(content, lines):
    findings = []

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            continue

        # ── Flatten directly before Dense ──
        if re.search(r"Flatten\(\)|\.flatten\(\)|tf\.reshape\(.+,\s*\[\s*-1\s*\]\)", s):
            for j in range(i + 1, min(len(lines), i + 4)):
                ns = lines[j].strip()
                if ns.startswith("#"):
                    continue
                if re.search(r"Dense\(|tf\.layers\.dense\(|tf\.keras\.layers\.Dense\(", ns):
                    nearby = "\n".join(lines[max(0, i - 5):i])
                    if not re.search(r"GlobalAveragePooling|GlobalMaxPooling|global_average_pooling|global_max_pooling", nearby):
                        findings.append((
                            i + 1, get_snippet(lines, i),
                            "Flatten() directly before Dense layer. Creates a "
                            "massive intermediate tensor and weight matrix. "
                            "GlobalAveragePooling2D/GlobalMaxPooling2D would "
                            "reduce dimensionality first, using far less memory."
                        ))
                    break
                if ns:
                    break

        # ── tf.concat self-accumulation in loop ──
        if re.search(r"tf\.concat\(", s):
            for k in range(i - 1, max(-1, i - 20), -1):
                pk = lines[k]
                ps = pk.strip()
                if re.match(r"(for|while)\s+", ps):
                    li = len(pk) - len(pk.lstrip())
                    ci = len(line) - len(line.lstrip())
                    if ci > li:
                        cm = re.match(r"(\w+)\s*=\s*tf\.concat\(\s*\[.*\b\1\b", s)
                        if cm:
                            findings.append((
                                i + 1, get_snippet(lines, i),
                                f"tf.concat() accumulates into '{cm.group(1)}' "
                                f"inside a loop. Tensor grows every iteration — "
                                f"previous allocations not freed. Use "
                                f"tf.TensorArray or list + single concat after loop."
                            ))
                    break

        # ── 3+ consecutive expand_dims ──
        if re.search(r"tf\.expand_dims\(", s):
            ct = 0
            for j in range(i, min(len(lines), i + 5)):
                if re.search(r"tf\.expand_dims\(", lines[j].strip()):
                    ct += 1
            if ct >= 3 and (i == 0 or not re.search(r"tf\.expand_dims\(", lines[i - 1].strip())):
                findings.append((
                    i + 1, get_snippet(lines, i, ct + 1),
                    f"{ct} consecutive tf.expand_dims() calls. Each allocates "
                    f"a new tensor copy. A single tf.reshape() to target shape "
                    f"would use 1 allocation instead of {ct}."
                ))

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Combined ML Code Smell Detector — Code Search Edition")
    print("=" * 60)
    print(f"Star filter: {MIN_STARS}–{MAX_STARS}")
    print(f"Token: {'set' if GITHUB_TOKEN else 'NOT SET (will hit low rate limits!)'}")

    init_all_csvs()
    totals = {"smell1": 0, "smell2": 0, "smell3": 0, "smell4": 0}

    smell_configs = [
        ("smell1", "Unbounded Graph Expansion", SMELL1_QUERIES, verify_smell1),
        ("smell2", "Graph-Constant Bottleneck",  SMELL2_QUERIES, verify_smell2),
        ("smell3", "GPU Released Memory Failure", SMELL3_QUERIES, verify_smell3),
        ("smell4", "Shape Mismatch Leak",         SMELL4_QUERIES, verify_smell4),
    ]

    for smell_key, label, queries, verify_fn in smell_configs:
        print(f"\n{'=' * 60}")
        print(f"Scanning for: {label}")
        print(f"  Queries: {len(queries)}")
        print(f"{'=' * 60}")

        count = search_and_verify(queries, verify_fn, smell_key)
        totals[smell_key] = count
        print(f"\n  → {label}: {count} findings saved to {OUTPUT_FILES[smell_key]}")

    print(f"\n{'=' * 60}")
    print("DONE")
    for key, count in totals.items():
        print(f"  {OUTPUT_FILES[key]}: {count} findings")
    print(f"  Total: {sum(totals.values())} findings")
    print(f"  Unique files scanned: {len(processed_files)}")
    print(f"  Repos checked for stars: {len(star_cache)}")


if __name__ == "__main__":
    main()
