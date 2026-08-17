#!/usr/bin/env python3
"""Diagnose GitHub access for the analysis pipeline.

Run this when a nightly run reports zero files. It checks the token, the rate
limit budget, every configured search query, and one real file download, so the
failure is attributable rather than guessed at.

    python scripts/check_github_access.py

Reads GITHUB_PAT from the environment or a local .env file. Exits non-zero if
anything the pipeline depends on is broken.
"""
import ast
import os
import sys

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_token():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    except ImportError:
        pass
    return os.getenv("GITHUB_PAT")


def search_queries():
    """Read the queries out of app.py without importing it (import has side effects)."""
    tree = ast.parse(open(os.path.join(REPO_ROOT, "app.py"), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "FILE_TYPES" for t in node.targets):
            return {
                key.value: dict(zip([k.value for k in cfg.keys],
                                    [getattr(v, "value", None) for v in cfg.values]))["search_query"]
                for key, cfg in zip(node.value.keys, node.value.values)
            }
    raise SystemExit("Could not find FILE_TYPES in app.py")


def main():
    token = load_token()
    if not token:
        print("FAIL  GITHUB_PAT is not set (checked environment and .env)")
        return 1
    print(f"OK    GITHUB_PAT is set ({len(token)} chars, starts {token[:4]}...)")

    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    failures = []

    r = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=30)
    if r.status_code == 401:
        print("FAIL  Token rejected (401). It is expired, revoked, or mistyped.")
        print("      Issue a new PAT and update GITHUB_PAT in the host's environment.")
        return 1
    if r.status_code != 200:
        print(f"FAIL  /rate_limit returned {r.status_code}: {r.text[:200]}")
        return 1

    resources = r.json().get("resources", {})
    for name in ("core", "code_search", "search"):
        q = resources.get(name)
        if q:
            print(f"OK    {name} quota: {q['remaining']}/{q['limit']} remaining")
            if q["remaining"] == 0:
                failures.append(f"{name} quota exhausted")

    sample_url = None
    for file_type, query in search_queries().items():
        r = requests.get("https://api.github.com/search/code", headers=headers, timeout=30,
                         params={"q": query, "per_page": 1, "sort": "indexed", "order": "desc"})
        if r.status_code != 200:
            print(f"FAIL  search '{query}' -> {r.status_code}: {r.text[:200]}")
            failures.append(f"{file_type} search failed ({r.status_code})")
            continue
        items = r.json().get("items", [])
        print(f"OK    search '{query}' -> {r.json().get('total_count', 0):,} matches")
        if not items:
            failures.append(f"{file_type} search returned no items")
        elif sample_url is None:
            sample_url = items[0].get("download_url")

    # Searching is not enough - the run also downloads each match.
    if sample_url:
        r = requests.get(sample_url, headers=headers, timeout=30)
        if r.status_code == 200:
            print(f"OK    downloaded a sample file ({len(r.text)} bytes)")
        else:
            print(f"FAIL  sample download -> {r.status_code}: {r.text[:200]}")
            failures.append("file download failed")

    if failures:
        print("\nProblems found:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed - the pipeline's GitHub access is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
