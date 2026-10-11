"""
embed_chunks.py
Create embeddings for chunks that do not have one yet, using a local Ollama model,
and store them in chunks.embedding (pgvector).

  - Picks chunks where embedding IS NULL (or was made by a different model).
    ACTIVE AND INACTIVE chunks are both embedded, so re-activating a test case later
    needs no re-embedding. Use --active-only to skip inactive ones.
    (Search still only returns active chunks: that filter lives in the search query.)
  - Sends "search_document: " + content   (this model expects a task prefix;
    questions at search time must use "search_query: ").
  - Saves after every batch, so you can stop and re-run safely.
  - Refuses to cut text silently: if a chunk is longer than the model's context,
    it is reported and left NULL (nothing wrong is stored).
  - At the end it prints the REAL characters-per-token of your data, so you can set
    --chars-per-token in testcase_ingest.py correctly.

Usage (PowerShell):
  $env:PG_DSN = "postgresql://postgres:yourpassword@localhost:5432/ragdb"
  python embed_chunks.py --limit 20            # small test first
  python embed_chunks.py                       # everything pending (active + inactive)
  python embed_chunks.py --active-only         # skip inactive chunks
  python embed_chunks.py --create-index        # also build the HNSW vector index at the end
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DOC_PREFIX = "search_document: "
EXPECTED_DIM = 768            # must match chunks.embedding VECTOR(768)
MAX_FAIL_STREAK = 30          # this many chunks failing in a row with no success = something is wrong


class EmbedError(Exception):
    pass


def ollama_embed(url, model, texts, timeout=300, retries=3):
    """Return (vectors, prompt_tokens). truncate=False: error instead of silently cutting."""
    body = json.dumps({"model": model, "input": texts, "truncate": False}).encode("utf-8")
    req = urllib.request.Request(f"{url}/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    data = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code == 404:                      # wrong model name: no point continuing
                raise SystemExit(f"Ollama says: {msg}\nCheck the model name with `ollama list`.")
            raise EmbedError(msg) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise SystemExit(f"Cannot reach Ollama at {url} after {retries} tries ({e}). Is it running?")
            time.sleep(2 * attempt)                # transient problem: wait and retry
    vectors = data.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise EmbedError(f"unexpected response from Ollama: {str(data)[:200]}")
    return vectors, data.get("prompt_eval_count")


def check_dim(vec):
    if len(vec) != EXPECTED_DIM:
        raise SystemExit(f"Model returned {len(vec)} numbers but the column is VECTOR({EXPECTED_DIM}). "
                         "Use the matching model or change the column size.")


def embed_with_fallback(url, model, rows, stats):
    """Try the whole batch; if it fails, try one by one so a single bad chunk
    does not block the others. Returns (good: [(row, vector)], failed: [(id, reason)])."""
    texts = [DOC_PREFIX + r[1] for r in rows]

    def count(texts_sent, tokens):
        if tokens:
            stats["tokens"] += tokens
            stats["chars"] += sum(len(t) for t in texts_sent)

    try:
        vectors, tokens = ollama_embed(url, model, texts)
        count(texts, tokens)
        return list(zip(rows, vectors)), []
    except EmbedError:
        pass
    good, failed = [], []
    for row, text in zip(rows, texts):
        try:
            vectors, tokens = ollama_embed(url, model, [text])
            count([text], tokens)
            good.append((row, vectors[0]))
        except EmbedError as e:
            failed.append((row[0], str(e)))
    return good, failed


# content_hash guard on SAVE: if the text changed while we were embedding, do not store a stale vector
SELECT_PENDING = """
SELECT id, content, content_hash FROM chunks
WHERE id > %(last)s {active}
  AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s)
ORDER BY id LIMIT %(n)s
"""
SAVE = """
UPDATE chunks SET embedding = %s::vector, embedding_model = %s
WHERE id = %s AND content_hash = %s
"""
COUNT_PENDING = """
SELECT count(*) FROM chunks
WHERE (embedding IS NULL OR embedding_model IS DISTINCT FROM %s) {active}
"""


def main():
    ap = argparse.ArgumentParser(description="Embed pending chunks with Ollama")
    ap.add_argument("--model", default="nomic-embed-text-v2-moe")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="stop after this many chunks (for testing)")
    ap.add_argument("--active-only", action="store_true", help="skip chunks that are not active")
    ap.add_argument("--create-index", action="store_true", help="build the HNSW index at the end")
    args = ap.parse_args()
    if "PG_DSN" not in os.environ:
        sys.exit("Set PG_DSN first, e.g.  $env:PG_DSN = 'postgresql://postgres:yourpassword@localhost:5432/ragdb'")

    import psycopg

    active = "AND is_active" if args.active_only else ""      # constant text, not user input
    select_sql = SELECT_PENDING.format(active=active)
    count_sql = COUNT_PENDING.format(active=active)

    done, stale, failed_all, fail_streak, last = 0, 0, [], 0, ""
    stats = {"tokens": 0, "chars": 0}
    started = time.time()
    with psycopg.connect(os.environ["PG_DSN"]) as conn:
        total = conn.execute(count_sql, (args.model,)).fetchone()[0]
        total = min(total, args.limit) if args.limit else total
        scope = "active chunks only" if args.active_only else "active and inactive chunks"
        print(f"{total} chunks to embed with {args.model} ({scope})")

        while True:
            n = args.batch_size if not args.limit else min(args.batch_size, args.limit - done - stale - len(failed_all))
            if n <= 0:
                break
            rows = conn.execute(select_sql, {"last": last, "model": args.model, "n": n}).fetchall()
            if not rows:
                break
            last = rows[-1][0]

            good, failed = embed_with_fallback(args.ollama_url, args.model, rows, stats)
            failed_all += failed
            fail_streak = 0 if good else fail_streak + len(failed)
            if fail_streak >= MAX_FAIL_STREAK:
                raise SystemExit(f"{fail_streak} chunks in a row failed. Last error: {failed[-1][1]}\nStopping.")
            if good:
                check_dim(good[0][1])
                with conn.cursor() as cur:
                    for row, vec in good:
                        cur.execute(SAVE, (json.dumps(vec), args.model, row[0], row[2]))
                        if cur.rowcount:
                            done += 1
                        else:
                            stale += 1                  # text changed (or row removed) meanwhile
                conn.commit()                           # progress is saved after every batch
            rate = (done + stale + len(failed_all)) / max(time.time() - started, 0.001)
            print(f"  embedded {done}/{total}   failed {len(failed_all)}   ({rate:.1f} chunks/s)", flush=True)

        if args.create_index:
            print("Building HNSW index ...")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vec ON chunks "
                         "USING hnsw (embedding vector_cosine_ops)")
            conn.commit()
        left = conn.execute(count_sql, (args.model,)).fetchone()[0]

    print(f"\nDone: {done} embedded, {len(failed_all)} failed, {stale} skipped (text changed meanwhile), "
          f"{left} still pending.")
    if stats["tokens"]:
        cpt = stats["chars"] / stats["tokens"]
        print(f"Measured on this run: {cpt:.2f} characters per token "
              f"(testcase_ingest.py assumes 3.5; use --chars-per-token {cpt:.1f} if it differs a lot).")
    for cid, reason in failed_all[:10]:
        print(f"  FAILED {cid}: {reason}")
    if len(failed_all) > 10:
        print(f"  ... {len(failed_all) - 10} more")
    if failed_all:
        print("Failed chunks are left NULL and are retried on the next run. 'exceeds the context length' "
              "means the chunk is too long: re-run testcase_ingest.py with a smaller --max-chunk-tokens "
              "or the measured --chars-per-token.")
        sys.exit(1)


if __name__ == "__main__":
    main()