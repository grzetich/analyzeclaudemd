"""Local dev launcher: overrides PERSISTENT_DATA_DIR to ./data and starts Flask."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import app  # noqa: E402

# Override the Render path so cached data is read from ./data locally.
app.PERSISTENT_DATA_DIR = os.path.join(HERE, "data")
os.makedirs(app.PERSISTENT_DATA_DIR, exist_ok=True)

# Re-run migration now that PERSISTENT_DATA_DIR points at the right place.
app.migrate_legacy_data_files()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=== Local dev server on http://127.0.0.1:{port} ===")
    print("Routes: /  /agents  /skills  /topic-evolution?type=...  /trends?type=...  /visualization?type=...")
    app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
