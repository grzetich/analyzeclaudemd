"""Regression tests for GitHub collection failure handling.

A failed request must never be reported as "no files found" - that is what hid
the 2026-08-17 outage. Runs offline against mocked responses:

    python test_collection.py
"""
import os, sys, unittest
from unittest import mock
os.environ.setdefault("GITHUB_PAT", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app

class Resp:
    def __init__(self, status, json_body=None, text="", headers=None, links=None):
        self.status_code, self._j, self.text = status, json_body or {}, text
        self.headers, self.links = headers or {}, links or {}
    def json(self): return self._j

def item(n): return {"download_url": f"https://raw/{n}", "path": f"{n}.md",
                     "repository": {"full_name": f"o/r{n}"}}

def run(responses):
    """responses: dict url-substring -> list of Resp (popped in order)"""
    def fake_get(url, headers=None, params=None, timeout=None):
        for key, queue in responses.items():
            if key in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected url {url}")
    with mock.patch.object(app.requests, "get", side_effect=fake_get), \
         mock.patch.object(app.time, "sleep"):
        return app.get_claude_md_files("filename:claude.md", {}, max_files=10)

class T(unittest.TestCase):
    def test_expired_token_names_the_cause(self):
        with self.assertRaises(app.GitHubCollectionError) as cm:
            run({"search/code": [Resp(401, text='{"message":"Bad credentials"}')]})
        self.assertIn("401", str(cm.exception)); self.assertIn("expired", str(cm.exception))
        self.assertEqual(cm.exception.status_code, 401)

    def test_401_is_not_retried(self):
        calls = []
        def fake_get(url, **kw): calls.append(url); return Resp(401, text="nope")
        with mock.patch.object(app.requests, "get", side_effect=fake_get), \
             mock.patch.object(app.time, "sleep"):
            with self.assertRaises(app.GitHubCollectionError): 
                app.get_claude_md_files("q", {}, max_files=10)
        self.assertEqual(len(calls), 1)

    def test_server_error_retries_then_fails(self):
        calls = []
        def fake_get(url, **kw): calls.append(url); return Resp(503, text="unavailable")
        with mock.patch.object(app.requests, "get", side_effect=fake_get), \
             mock.patch.object(app.time, "sleep"):
            with self.assertRaises(app.GitHubCollectionError) as cm:
                app.get_claude_md_files("q", {}, max_files=10)
        self.assertEqual(len(calls), app.SEARCH_MAX_ATTEMPTS)
        self.assertIn("503", str(cm.exception))

    def test_server_error_then_success_recovers(self):
        docs = run({"search/code": [Resp(503, text="x"), Resp(200, {"items": [item(1)]})],
                    "raw/": [Resp(200, text="hello world")]})
        self.assertEqual(docs, ["hello world"])

    def test_genuine_empty_result_is_not_an_error(self):
        self.assertEqual(run({"search/code": [Resp(200, {"items": []})]}), [])

    def test_all_downloads_failing_is_reported_as_failure(self):
        with self.assertRaises(app.GitHubCollectionError) as cm:
            run({"search/code": [Resp(200, {"items": [item(1), item(2)]})],
                 "raw/": [Resp(404, text="")]})
        self.assertIn("all 2 downloads failed", str(cm.exception))

    def test_partial_results_survive_a_later_page_failure(self):
        docs = run({"search/code": [Resp(200, {"items": [item(1)]}, links={"next": {}}),
                                    Resp(401, text="expired mid-run")],
                    "raw/": [Resp(200, text="doc one")]})
        self.assertEqual(docs, ["doc one"])

    def test_rate_limit_waits_then_proceeds(self):
        import time as real_time
        reset = int(real_time.time()) + 10
        docs = run({"search/code": [Resp(403, headers={"X-RateLimit-Remaining": "0",
                                                       "X-RateLimit-Reset": str(reset)}),
                                    Resp(200, {"items": [item(1)]})],
                    "raw/": [Resp(200, text="after wait")]})
        self.assertEqual(docs, ["after wait"])

    def test_long_rate_limit_wait_fails_fast(self):
        import time as real_time
        reset = int(real_time.time()) + 4000
        with self.assertRaises(app.GitHubCollectionError) as cm:
            run({"search/code": [Resp(403, headers={"X-RateLimit-Remaining": "0",
                                                    "X-RateLimit-Reset": str(reset)})]})
        self.assertIn("longer than", str(cm.exception))

    def test_403_with_quota_left_blames_permissions(self):
        with self.assertRaises(app.GitHubCollectionError) as cm:
            run({"search/code": [Resp(403, headers={"X-RateLimit-Remaining": "42"}, text="forbidden")]})
        self.assertIn("public repository access", str(cm.exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)
