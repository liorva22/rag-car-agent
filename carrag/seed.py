"""Build the Postgres index from data/corpus.jsonl.gz (+ committed embeddings). No-op if it's already loaded.

python -m carrag.seed [--force]
"""

import sys

from . import store
from .corpus import load_corpus


def main(argv=None) -> int:
    conn = store.connect()
    n, want = store.count(conn), len(load_corpus())
    if n == want and "--force" not in (argv or sys.argv[1:]):
        print(f"index already has {n} docs")
        return 0
    store.load(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
