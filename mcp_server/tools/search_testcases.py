import json
import os
import urllib.error
import urllib.request
from typing import Any

import psycopg
from psycopg import sql as pg_sql
from mcp.server.fastmcp import FastMCP


EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EXPECTED_DIMENSIONS = 768
MAX_LIMIT = 100

mcp = FastMCP("qa-testcase-search")


def _embed_query(query: str) -> list[float]:
    request_body = json.dumps({
        "model": EMBED_MODEL,
        "input": [f"search_query: {query}"],
        "truncate": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Ollama embedding request failed: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc

    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise RuntimeError("Ollama returned an invalid embedding response")

    vector = vectors[0]
    if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSIONS:
        actual = len(vector) if isinstance(vector, list) else "unknown"
        raise RuntimeError(
            f"{EMBED_MODEL} returned {actual} dimensions; "
            f"the database requires {EXPECTED_DIMENSIONS}"
        )
    return vector


@mcp.tool()
def search_testcases(
    query: str,
    module: str | None = None,
    high_level: bool | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find test cases by semantic similarity, with optional module filters.

    Args:
        query: Natural-language description of the test case to find.
        module: Optional module name, matched case-insensitively.
        high_level: Optional filter for high-level test cases.
        limit: Maximum number of test cases to return (1-100).

    Returns:
        Matching test cases with their best matching chunk and similarity score.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > 2000:
        raise ValueError("query must be at most 2000 characters")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    if module is not None and not isinstance(module, str):
        raise ValueError("module must be a string or null")

    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("Set PG_DSN before starting the MCP server")

    query_vector = _embed_query(query.strip())
    filters = [
        pg_sql.SQL("c.source_type = 'testcase'"),
        pg_sql.SQL("c.embedding IS NOT NULL"),
        pg_sql.SQL("c.embedding_model = %(model)s"),
    ]
    parameters: dict[str, Any] = {
        "query_vector": json.dumps(query_vector),
        "model": EMBED_MODEL,
        "limit": limit,
    }
    if module and module.strip():
        filters.append(pg_sql.SQL("lower(td.module) = lower(%(module)s)"))
        parameters["module"] = module.strip()
    if high_level is not None:
        filters.append(pg_sql.SQL("td.high_level = %(high_level)s"))
        parameters["high_level"] = high_level

    statement = pg_sql.SQL("""
        WITH best_chunks AS (
            SELECT DISTINCT ON (d.doc_id)
                d.doc_id,
                d.title,
                td.module,
                td.section,
                td.tc_label,
                td.tc_number,
                td.preconditions,
                td.scenario,
                td.steps,
                td.test_data,
                td.expected,
                c.id AS chunk_id,
                c.content AS matched_text,
                1 - (c.embedding <=> %(query_vector)s::vector) AS similarity
            FROM chunks AS c
            JOIN documents AS d ON d.doc_id = c.doc_id
            JOIN testcase_details AS td ON td.doc_id = d.doc_id
            WHERE {}
            ORDER BY d.doc_id, c.embedding <=> %(query_vector)s::vector
        )
        SELECT *
        FROM best_chunks
        ORDER BY similarity DESC
        LIMIT %(limit)s
    """).format(pg_sql.SQL(" AND ").join(filters))

    with psycopg.connect(dsn) as conn:
        cursor = conn.execute(statement, parameters)
        columns = [description.name for description in cursor.description or []]
        rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]
