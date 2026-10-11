"""
source_code_ingest.py

Read the NOVA_AUTO_Prithvi source tree and load it into PostgreSQL (schema_v3.sql):

    one code symbol  ->  1 row in documents
    (method, class,      1 row in source_code_details   (same doc_id)
     function, test,     1..n rows in chunks            (split only if text is too long)
     table, API path,
     README section)

Chunking (see nova-data.md):
  - Java / TypeScript : tree-sitter syntax tree -> one document per method / function /
                        test; one outline document per class (method bodies removed)
  - OpenAPI yaml      : one document per path, one per component schema
  - SQL               : one document per CREATE TABLE (INSERT/UPDATE data rows are dropped)
  - Markdown          : one document per heading section
  - anything else     : the whole file, split by lines
  - every chunk starts with a short header (service / file / symbol) so a bare method
    body is still searchable; long units are split at line breaks, counted with the
    embedding model's tokenizer, and the header is repeated in every part.

Sensitive files (keystores, env files, application-local/test yaml, credential classes,
seed SQL, ...) are never read. Secret-looking values inside the files that ARE read are
replaced with <REDACTED> before anything is stored.

Embeddings are NOT created here: chunks.embedding stays NULL until embed_chunks.py runs.

Safe to re-run (upsert):
  - unchanged chunk  -> keeps its embedding
  - changed chunk    -> embedding reset to NULL (re-embedded next run)
  - symbols/files that disappeared -> is_active = FALSE
Everything is written in ONE transaction: all or nothing.

Usage (PowerShell):
  pip install "psycopg[binary]" tokenizers huggingface_hub tree-sitter tree-sitter-java tree-sitter-typescript
  $env:PG_DSN = "postgresql://postgres:<password>@localhost:5432/ragdb"
  python source_code_ingest.py --dry-run                         # check first
  python source_code_ingest.py                                   # load everything
  python source_code_ingest.py --only-service nova-medical       # one service
  python source_code_ingest.py --include-infra                   # + CloudFormation / CI yaml
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


SOURCE_TYPE = "source_code"

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "NOVA_AUTO_Prithvi"
DEFAULT_MAX_CHUNK_TOKENS = 400
DEFAULT_TOKENIZER_REPO = "nomic-ai/nomic-embed-text-v2-moe"
DEFAULT_EMBEDDING_PREFIX = "search_document: "
MIN_CHUNK_TOKENS = 32
MIN_UNIT_CHARS = 20
MAX_FILE_BYTES = 1_000_000

# ---------- which files are read ----------

EXCLUDE_DIRS = {
    "target", "node_modules", "dist", "build", "coverage", ".git", ".husky", ".vscode",
    ".mvn", ".idea", ".auth", ".storybook", "public", "assets", "stories", "keycloak-theme",
    "allure-report", "allure-results", "playwright-report", "test-results", "logs",
    "generated-sources", "generated-test-sources", "__pycache__",
}

# Never opened. Only the file name is counted in the summary.
SENSITIVE_NAME = re.compile(
    r"(\.(p12|pfx|jks|pem|key|crt|cer|mdmp)$)"
    r"|(^\.env)|(\.env(\.|$))|(^env\.)"
    r"|(^application-(local|test)[\w.-]*\.ya?ml$)"
    r"|(^\.npmrc$)"
    r"|(credential)|(secret)|(keystore)|(truststore)"
    r"|(seed)",
    re.I,
)

LANG_BY_EXT = {".java": "java", ".ts": "typescript", ".tsx": "tsx", ".yaml": "yaml",
               ".yml": "yaml", ".sql": "sql", ".md": "markdown", ".sh": "bash"}


def classify(rel, include_infra):
    """-> (language, layer) or None when the file should not be ingested."""
    parts = rel.split("/")
    name = parts[-1]
    ext = Path(name).suffix.lower()
    lang = LANG_BY_EXT.get(ext)
    if lang is None:
        return None
    padded = "/" + rel

    if ext == ".java":
        if "/src/main/java/" in padded and name != "package-info.java":
            return lang, "java"
        return None
    if ext in (".ts", ".tsx"):
        if name.endswith((".d.ts", ".test.ts", ".test.tsx", ".stories.tsx", ".spec.tsx")):
            return None
        if parts[0] == "nova-web" and len(parts) > 1 and parts[1] == "src" and "test" not in parts:
            return lang, "web"
        if parts[0] == "qe-pulse-2-0" and len(parts) > 1 and parts[1] in ("src", "tests", "scripts"):
            return lang, "qe"
        return None
    if ext in (".yaml", ".yml"):
        if "/api-spec/" in padded:
            return "openapi", "api-spec"
        if include_infra and (parts[0] == "infrastructure" or "/ci/aws/deploys/" in padded
                              or name == ".gitlab-ci.yml"):
            return "yaml", "infra"
        return None
    if ext == ".sql":
        low = name.lower()
        if "/mysql/init/" in padded and "schema" in low and "/test/resources/" not in padded:
            return lang, "sql"
        return None
    if ext == ".md":
        return lang, "docs"
    if ext == ".sh":
        if "/docker/test/" in padded:
            return lang, "test-script"
        return None
    return None


def discover(root, services, include_infra):
    """Walk the tree -> (files to read, sensitive files that were skipped)."""
    files, sensitive = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if services and rel.split("/")[0] not in services:
                continue
            if SENSITIVE_NAME.search(fn):
                sensitive.append(rel)
                continue
            info = classify(rel, include_infra)
            if info:
                files.append({"path": full, "rel": rel, "language": info[0], "layer": info[1]})
    return files, sensitive


# ---------- small helpers ----------

def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def tidy(text):
    """Tabs -> spaces, strip line ends, at most one blank line in a row."""
    text = text.replace("\r\n", "\n").replace("\t", "  ")
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = []
    for ln in lines:
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return "\n".join(out).strip()


# ---------- secret redaction (values only; keys stay) ----------

_KEY = r"(?:password|passwd|pwd|secret|token|api[_-]?key|private[_-]?key|access[_-]?key|credential)s?"
RE_QUOTED = re.compile(rf"(?i)(\b[\w.-]*{_KEY}[\w.-]*\s*[:=]\s*)([\"'])([^\"'\n]{{3,}})\2")
RE_PLAIN = re.compile(rf"(?im)^(\s*[\w.-]*{_KEY}[\w.-]*\s*[:=]\s*)(?![\"'<$\{{])([^\s#][^\n#]*)$")
RE_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}")
RE_AWS = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")
RE_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)", re.S)
RE_URL_PW = re.compile(r"(://[^/\s:@]+:)([^@\s/]+)(@)")


def redact(text, language):
    """-> (text with secret values replaced, number of replacements)."""
    total = 0
    text, n = RE_QUOTED.subn(lambda m: f"{m.group(1)}{m.group(2)}<REDACTED>{m.group(2)}", text)
    total += n
    if language in ("yaml", "openapi", "markdown", "bash"):
        text, n = RE_PLAIN.subn(lambda m: f"{m.group(1)}<REDACTED>", text)
        total += n
    for rx, repl in ((RE_JWT, "<REDACTED>"), (RE_AWS, "<REDACTED>"), (RE_PEM, "<REDACTED PRIVATE KEY>"),
                     (RE_URL_PW, r"\1<REDACTED>\3")):
        text, n = rx.subn(repl, text)
        total += n
    return text, total


# ---------- tree-sitter (Java / TypeScript) ----------

_PARSERS = {}


def get_parser(language):
    """Return a tree-sitter parser, or None when the packages are not installed."""
    if language in _PARSERS:
        return _PARSERS[language]
    parser = None
    try:
        from tree_sitter import Language, Parser
        if language == "java":
            import tree_sitter_java as mod
            lang = Language(mod.language())
        elif language == "typescript":
            import tree_sitter_typescript as mod
            lang = Language(mod.language_typescript())
        else:
            import tree_sitter_typescript as mod
            lang = Language(mod.language_tsx())
        parser = Parser(lang)
    except Exception:
        parser = None
    _PARSERS[language] = parser
    return parser


def node_text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def field_text(node, field, src):
    child = node.child_by_field_name(field)
    return node_text(child, src) if child is not None else None


def lead_comment(node, src):
    """The /** doc comment directly above a node, if any."""
    prev = node.prev_sibling
    if (prev is not None and prev.type in ("block_comment", "comment")
            and node.start_point[0] - prev.end_point[0] <= 1
            and node_text(prev, src).startswith("/**")):
        return prev
    return None


def make_unit(node, src, name, kind, signature=None, text=None):
    lead = lead_comment(node, src)
    first = lead if lead is not None else node
    col = first.start_point[1]
    if text is None:
        text = src[first.start_byte:node.end_byte].decode("utf-8", "replace")
    elif lead is not None:
        text = node_text(lead, src) + "\n" + text
    lines = text.split("\n")
    lines = lines[:1] + [re.sub(rf"^[ \t]{{0,{col}}}", "", ln) for ln in lines[1:]]
    return {"name": name, "kind": kind, "signature": signature,
            "text": "\n".join(lines),
            "start_line": first.start_point[0] + 1, "end_line": node.end_point[0] + 1}


# ----- Java -----

JAVA_TYPES = {"class_declaration", "interface_declaration", "enum_declaration",
              "record_declaration", "annotation_type_declaration"}
JAVA_METHODS = {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
ACCESSOR = re.compile(r"^(get|set|is)[A-Z]")


def java_members(body):
    for c in body.children:
        if c.type == "enum_body_declarations":
            yield from c.children
        else:
            yield c


def java_header(node, body, src):
    return clean_ws(src[node.start_byte:body.start_byte].decode("utf-8", "replace")) if body is not None \
        else clean_ws(node_text(node, src))


def java_outline(node, body, src):
    """Class text with every method body removed: fields + method signatures only."""
    parts = []
    for c in java_members(body):
        if c.type in JAVA_METHODS:
            parts.append(java_header(c, c.child_by_field_name("body"), src) + ";")
        elif c.type in JAVA_TYPES:
            parts.append(java_header(c, c.child_by_field_name("body"), src) + " { ... }")
        elif c.type in ("{", "}", ";"):
            continue
        else:
            parts.append(node_text(c, src))
    return java_header(node, body, src) + " {\n  " + "\n  ".join(parts) + "\n}"


def java_units(root, src):
    units = []

    def visit(container, scope):
        for c in java_members(container):
            if c.type in JAVA_TYPES:
                name = field_text(c, "name", src) or "?"
                qual = scope + [name]
                body = c.child_by_field_name("body")
                kind = c.type.replace("_declaration", "")
                if body is not None:
                    units.append(make_unit(c, src, ".".join(qual), kind,
                                           java_header(c, body, src), java_outline(c, body, src)))
                    visit(body, qual)
            elif c.type in JAVA_METHODS:
                body = c.child_by_field_name("body")
                if body is None:                      # abstract / interface method: already in the outline
                    continue
                name = field_text(c, "name", src) or "?"
                if ACCESSOR.match(name) and body.end_point[0] - body.start_point[0] <= 3:
                    continue                          # trivial getter/setter: noise
                kind = "constructor" if c.type != "method_declaration" else "method"
                units.append(make_unit(c, src, ".".join(scope + [name]), kind,
                                       java_header(c, body, src)))
    visit(root, [])
    return units


# ----- TypeScript / TSX -----

TEST_CALL = re.compile(r"^(test|it)(\.(only|skip|fixme|slow|fail))?$")
DESCRIBE_CALL = re.compile(r"^(test\.)?describe(\.\w+)*$")
HOOK_CALL = re.compile(r"^(test\.)?(beforeAll|beforeEach|afterAll|afterEach)$")
FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}


def first_string_arg(args, src):
    for c in args.named_children:
        if c.type == "string":
            return node_text(c, src).strip("'\"")
        if c.type == "template_string":
            return node_text(c, src).strip("`")
        break
    return None


def ts_outline(node, body, src):
    parts = []
    for c in body.named_children:
        if c.type == "method_definition":
            b = c.child_by_field_name("body")
            sig = src[c.start_byte:b.start_byte].decode("utf-8", "replace") if b is not None else node_text(c, src)
            parts.append(clean_ws(sig) + ";")
        elif c.type != "comment":
            parts.append(node_text(c, src))
    head = clean_ws(src[node.start_byte:body.start_byte].decode("utf-8", "replace"))
    return head + " {\n  " + "\n  ".join(parts) + "\n}"


def ts_units(root, src):
    units = []

    def handle(node, scope):
        outer = node                                  # keeps `export` in the unit text
        if node.type == "export_statement":
            inner = node.child_by_field_name("declaration") or node.child_by_field_name("value")
            if inner is None:
                return
            node = inner
        t = node.type

        if t in ("function_declaration", "generator_function_declaration"):
            name = field_text(node, "name", src) or "default"
            body = node.child_by_field_name("body")
            sig = clean_ws(src[node.start_byte:body.start_byte].decode("utf-8", "replace")) if body else None
            units.append(make_unit(outer, src, ".".join(scope + [name]), "function", sig))

        elif t in ("class_declaration", "abstract_class_declaration"):
            name = field_text(node, "name", src) or "default"
            body = node.child_by_field_name("body")
            qual = scope + [name]
            if body is not None:
                units.append(make_unit(outer, src, ".".join(qual), "class",
                                       clean_ws(src[node.start_byte:body.start_byte].decode("utf-8", "replace")),
                                       ts_outline(node, body, src)))
                for m in body.named_children:
                    if m.type == "method_definition":
                        mb = m.child_by_field_name("body")
                        if mb is not None and mb.end_point[0] - mb.start_point[0] >= 2:
                            msig = clean_ws(src[m.start_byte:mb.start_byte].decode("utf-8", "replace"))
                            units.append(make_unit(m, src, ".".join(qual + [field_text(m, "name", src) or "?"]),
                                                   "method", msig))

        elif t in ("interface_declaration", "type_alias_declaration", "enum_declaration"):
            name = field_text(node, "name", src) or "?"
            units.append(make_unit(outer, src, ".".join(scope + [name]), t.replace("_declaration", "")
                                   .replace("type_alias", "type")))

        elif t in ("lexical_declaration", "variable_declaration"):
            for d in node.named_children:
                if d.type != "variable_declarator":
                    continue
                val = d.child_by_field_name("value")
                if val is None or val.end_point[0] - val.start_point[0] < 2:
                    continue
                name = field_text(d, "name", src) or "?"
                kind = "function" if val.type in FUNC_VALUES else "const"
                units.append(make_unit(outer, src, ".".join(scope + [name]), kind))
                break                                 # one unit per statement (it is one text range)

        elif t == "expression_statement":
            call = node.named_children[0] if node.named_children else None
            if call is None or call.type != "call_expression":
                return
            callee = (field_text(call, "function", src) or "").strip()
            args = call.child_by_field_name("arguments")
            if args is None:
                return
            title = first_string_arg(args, src)
            if DESCRIBE_CALL.match(callee):
                cb = args.named_children[-1] if args.named_children else None
                body = cb.child_by_field_name("body") if cb is not None and cb.type in FUNC_VALUES else None
                if body is not None:
                    for stmt in body.named_children:
                        handle(stmt, scope + [title or "describe"])
            elif TEST_CALL.match(callee) and title:
                units.append(make_unit(node, src, ".".join(scope + [title]), "test"))
            elif HOOK_CALL.match(callee):
                units.append(make_unit(node, src, ".".join(scope + [callee]), "hook"))

    for child in root.children:
        handle(child, [])
    return units


# ---------- non-code files ----------

def openapi_units(text):
    """Split an OpenAPI yaml: one block per path, per component, per top-level key."""
    blocks = []
    cur = None
    section = None
    start_line = 1
    for i, line in enumerate(text.split("\n"), 1):
        m = re.match(r"^(\s*)(\S[^:]*?):(\s|$)", line)
        indent = len(line) - len(line.lstrip(" "))
        new = None
        if m and not line.lstrip().startswith("#"):
            key = m.group(2).strip("'\"")
            if indent == 0:
                section, new = key, key
            elif section == "paths" and indent == 2:
                new = key
            elif section == "components" and indent in (2, 4):
                new = f"components.{key}" if indent == 2 else key
        if new is not None:
            cur = {"name": new, "lines": [], "start": i}
            blocks.append(cur)
        if cur is None:
            cur = {"name": "header", "lines": [], "start": i}
            blocks.append(cur)
        cur["lines"].append(line)
    units = []
    for b in blocks:
        body = "\n".join(b["lines"]).strip()
        if len([ln for ln in b["lines"] if ln.strip()]) < 2:
            continue                                  # a bare "components:" / "schemas:" line
        kind = "path" if b["name"].startswith("/") else "schema"
        units.append({"name": b["name"], "kind": kind, "signature": None, "text": body,
                      "start_line": b["start"], "end_line": b["start"] + len(b["lines"]) - 1})
    return units


RE_CREATE_TABLE = re.compile(r"create\s+table\s+(?:if\s+not\s+exists\s+)?[`\"]?([\w.]+)[`\"]?", re.I)
RE_DATA_STMT = re.compile(r"^\s*(insert|replace|update|delete)\b", re.I)


def sql_units(text):
    """One unit per CREATE TABLE. Data statements are dropped; other DDL goes into one unit."""
    units, other = [], []
    line_no = 1
    for stmt in re.split(r";[ \t]*\n", text):
        start = line_no
        line_no += stmt.count("\n") + 1
        body = re.sub(r"^(\s*--[^\n]*\n)+", "", stmt.strip())
        if not body or RE_DATA_STMT.match(body):
            continue
        m = RE_CREATE_TABLE.search(body)
        if m:
            units.append({"name": m.group(1), "kind": "table", "signature": None,
                          "text": body + ";", "start_line": start, "end_line": start + stmt.count("\n")})
        elif len(body) > 10:
            other.append(body + ";")
    if other:
        units.append({"name": "other_ddl", "kind": "ddl", "signature": None, "text": "\n".join(other),
                      "start_line": 1, "end_line": line_no})
    return units


def markdown_units(text):
    """One unit per heading section; very short sections are merged into the previous one."""
    sections = []
    cur = {"name": "intro", "lines": [], "start": 1}
    in_code = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,3})\s+(.*\S)\s*$", line)
        if m:
            sections.append(cur)
            cur = {"name": m.group(2), "lines": [], "start": i}
        cur["lines"].append(line)
    sections.append(cur)
    merged = []
    for s in sections:
        body = "\n".join(s["lines"]).strip()
        if not body:
            continue
        if merged and len(body) < 120:
            merged[-1]["text"] += "\n\n" + body
            merged[-1]["end_line"] = s["start"] + len(s["lines"]) - 1
            continue
        merged.append({"name": s["name"], "kind": "section", "signature": None, "text": body,
                       "start_line": s["start"], "end_line": s["start"] + len(s["lines"]) - 1})
    return merged


def file_unit(text, name="file"):
    return {"name": name, "kind": "file", "signature": None, "text": text,
            "start_line": 1, "end_line": text.count("\n") + 1}


STRATEGY_BY_LAYER = {"java": "ast_symbol", "web": "ast_symbol", "qe": "ast_symbol",
                     "api-spec": "openapi_block", "sql": "sql_table", "docs": "md_heading"}


def extract_units(f, text, warnings):
    """File text -> (units, strategy name)."""
    lang, layer = f["language"], f["layer"]
    if lang in ("java", "typescript", "tsx"):
        parser = get_parser(lang)
        if parser is None:
            warnings.append(f"{f['rel']}: tree-sitter not installed -> whole-file split")
            return [file_unit(text)], "file_window"
        src = text.encode("utf-8")
        tree = parser.parse(src)
        units = java_units(tree.root_node, src) if lang == "java" else ts_units(tree.root_node, src)
        if not units:
            return [file_unit(text)], "file_window"
        return units, "ast_symbol"
    if lang == "openapi":
        units = openapi_units(text)
    elif lang == "sql":
        units = sql_units(text)
    elif lang == "markdown":
        units = markdown_units(text)
    else:
        units = []
    if not units:
        return [file_unit(text)], "file_window"
    return units, STRATEGY_BY_LAYER[layer]


# ---------- chunking (one chunk normally; split only when too long) ----------

def count_tokens(text, tokenizer, prefix):
    return len(tokenizer.encode(prefix + text, add_special_tokens=True).ids)


def split_line(line, budget, tokenizer, prefix):
    """Split one very long line so every piece fits in `budget` tokens."""
    if count_tokens(line, tokenizer, prefix) <= budget:
        return [line]
    pieces, rest = [], line
    while rest:
        if count_tokens(rest, tokenizer, prefix) <= budget:
            pieces.append(rest)
            break
        lo, hi, best = 1, len(rest), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(rest[:mid], tokenizer, prefix) <= budget:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        cut = best
        if best < len(rest):
            boundary = max(rest.rfind(" ", 0, best), rest.rfind(",", 0, best))
            if boundary > best * 0.5:
                cut = boundary + 1
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return pieces


def build_chunks(header, body, max_tokens, tokenizer, prefix):
    """Whole text in one chunk if it fits; otherwise line-packed parts, header repeated."""
    full = f"{header}\n{body}"
    if count_tokens(full, tokenizer, prefix) <= max_tokens:
        return [full]
    marker = "(part 999999 of 999999)"
    overhead = count_tokens(f"{header}\n{marker}\n", tokenizer, prefix)
    budget = max_tokens - overhead
    if budget < 30:
        raise ValueError("Chunk header alone is too long for --max-chunk-tokens; raise the limit.")
    for _ in range(8):                                # shrink the budget until every part really fits
        lines = []
        for ln in body.split("\n"):
            lines.extend(split_line(ln, budget, tokenizer, prefix))
        parts, cur, cur_tokens = [], [], 0
        for ln in lines:
            t = count_tokens(ln, tokenizer, prefix) + 1
            if cur and cur_tokens + t > budget:
                parts.append(cur)
                cur, cur_tokens = [], 0
            cur.append(ln)
            cur_tokens += t
        if cur:
            parts.append(cur)
        total = len(parts)
        texts = [f"{header}\n(part {i} of {total})\n" + "\n".join(p) for i, p in enumerate(parts, 1)]
        if all(count_tokens(x, tokenizer, prefix) <= max_tokens for x in texts):
            return texts
        budget = int(budget * 0.9)
    raise ValueError("Could not split a unit within the token limit.")


# ---------- one file -> records for the 3 tables ----------

def feature_of(rel, layer):
    parts = rel.split("/")
    if layer == "java":
        m = re.search(r"/com/connexin/nova/[^/]+/([^/]+)/", "/" + rel)
        return m.group(1) if m else None
    if layer == "api-spec":
        return re.sub(r"^nova-[a-z]+-", "", Path(rel).stem)
    if layer == "web" and len(parts) > 3:
        return parts[3] if parts[2] == "domains" and len(parts) > 4 else parts[2]
    if layer == "qe" and len(parts) > 2:
        return parts[2] if parts[1] in ("tests", "src") and len(parts) > 3 else None
    return None


def build_records(f, text, repo, commit_sha, seen_ids, stats, warnings, max_tokens, tokenizer, prefix):
    rel, language, layer = f["rel"], f["language"], f["layer"]
    service = rel.split("/")[0]
    feature = feature_of(rel, layer)
    units, strategy = extract_units(f, text, warnings)

    documents, details, chunks = [], [], []
    for u in units:
        body, n_redacted = redact(tidy(u["text"]), language)
        stats["redacted"] += n_redacted
        if len(body) < MIN_UNIT_CHARS:
            continue

        base = f"code:{repo}:{rel}#{u['name']}"
        seen_ids[base] += 1
        doc_id = base if seen_ids[base] == 1 else f"{base}~{seen_ids[base]}"   # overloads / repeated names

        header = f"Service: {service} | Layer: {layer} | File: {rel}"
        if u["kind"] != "file":
            header += f"\n{u['kind'].capitalize()}: {u['name']}"
        if u["signature"]:
            header += f"\nSignature: {redact(u['signature'], language)[0]}"

        meta = {"service": service, "layer": layer, "feature": feature, "language": language,
                "symbol_kind": u["kind"]}
        documents.append({
            "doc_id": doc_id, "source_type": SOURCE_TYPE, "title": f"{u['name']} ({u['kind']})"[:200],
            "content": body, "content_hash": sha256(body), "metadata": meta,
        })
        details.append({
            "doc_id": doc_id, "repo": repo, "file_path": rel, "language": language,
            "symbol_name": u["name"], "symbol_kind": u["kind"],
            "signature": redact(u["signature"], language)[0] if u["signature"] else None,
            "start_line": u["start_line"], "end_line": u["end_line"], "commit_sha": commit_sha,
        })

        texts = build_chunks(header, body, max_tokens, tokenizer, prefix)
        for i, ctext in enumerate(texts):
            chunks.append({
                "id": f"{doc_id}:c{i}", "doc_id": doc_id, "source_type": SOURCE_TYPE, "chunk_index": i,
                "content": ctext, "content_hash": sha256(ctext),
                "chunk_strategy": strategy if len(texts) == 1 else strategy + "+line_split",
                "token_count": count_tokens(ctext, tokenizer, prefix),
                "metadata": {"part_index": i + 1, "part_count": len(texts), **meta},
            })
    return documents, details, chunks


# ---------- SQL ----------

UPSERT_DOCUMENT = """
INSERT INTO documents (doc_id, source_type, title, content, content_hash, metadata)
VALUES (%(doc_id)s, %(source_type)s, %(title)s, %(content)s, %(content_hash)s, %(metadata)s)
ON CONFLICT (doc_id) DO UPDATE SET
    title = EXCLUDED.title, content = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash, metadata = EXCLUDED.metadata,
    is_active = TRUE, ingested_at = now()
"""

UPSERT_DETAILS = """
INSERT INTO source_code_details (doc_id, repo, file_path, language, symbol_name, symbol_kind,
    signature, start_line, end_line, commit_sha)
VALUES (%(doc_id)s, %(repo)s, %(file_path)s, %(language)s, %(symbol_name)s, %(symbol_kind)s,
    %(signature)s, %(start_line)s, %(end_line)s, %(commit_sha)s)
ON CONFLICT (doc_id) DO UPDATE SET
    repo = EXCLUDED.repo, file_path = EXCLUDED.file_path, language = EXCLUDED.language,
    symbol_name = EXCLUDED.symbol_name, symbol_kind = EXCLUDED.symbol_kind,
    signature = EXCLUDED.signature, start_line = EXCLUDED.start_line,
    end_line = EXCLUDED.end_line, commit_sha = EXCLUDED.commit_sha
"""

UPSERT_CHUNK = """
INSERT INTO chunks (id, doc_id, source_type, chunk_index, content, content_hash,
                    chunk_strategy, token_count, metadata)
VALUES (%(id)s, %(doc_id)s, %(source_type)s, %(chunk_index)s, %(content)s, %(content_hash)s,
        %(chunk_strategy)s, %(token_count)s, %(metadata)s)
ON CONFLICT (id) DO UPDATE SET
    chunk_index = EXCLUDED.chunk_index, content = EXCLUDED.content,
    chunk_strategy = EXCLUDED.chunk_strategy, token_count = EXCLUDED.token_count,
    metadata = EXCLUDED.metadata, is_active = TRUE,
    embedding       = CASE WHEN chunks.content_hash = EXCLUDED.content_hash
                           THEN chunks.embedding ELSE NULL END,
    embedding_model = CASE WHEN chunks.content_hash = EXCLUDED.content_hash
                           THEN chunks.embedding_model ELSE NULL END,
    content_hash = EXCLUDED.content_hash
"""

# symbols/files of this repo (inside the scanned folders) that are gone now
DEACTIVATE_DOCS = """
UPDATE documents d SET is_active = FALSE
FROM source_code_details s
WHERE s.doc_id = d.doc_id AND s.repo = %s AND d.is_active
  AND EXISTS (SELECT 1 FROM unnest(%s::text[]) p WHERE starts_with(s.file_path, p))
  AND NOT (d.doc_id = ANY(%s))
"""

DEACTIVATE_STALE_CHUNKS = """
UPDATE chunks SET is_active = FALSE
WHERE is_active AND doc_id = ANY(%s) AND NOT (id = ANY(%s))
"""

DEACTIVATE_ORPHAN_CHUNKS = """
UPDATE chunks c SET is_active = FALSE
FROM documents d WHERE c.doc_id = d.doc_id AND NOT d.is_active AND c.is_active
"""

SUMMARY = """
SELECT (SELECT count(*) FROM documents WHERE source_type = 'source_code' AND is_active),
       (SELECT count(*) FROM source_code_details s JOIN documents d USING (doc_id) WHERE d.is_active),
       (SELECT count(*) FROM chunks WHERE source_type = 'source_code' AND is_active),
       (SELECT count(*) FROM chunks WHERE source_type = 'source_code' AND is_active AND embedding IS NULL),
       (SELECT count(*) FROM chunks WHERE source_type = 'source_code' AND NOT is_active)
"""


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def write_to_postgres(documents, details, chunks, repo, scope_prefixes, batch_size):
    import psycopg
    from psycopg.types.json import Jsonb

    for d in documents:
        d["metadata"] = Jsonb(d["metadata"])
    for c in chunks:
        c["metadata"] = Jsonb(c["metadata"])

    doc_ids = [d["doc_id"] for d in documents]
    chunk_ids = [c["id"] for c in chunks]

    with psycopg.connect(os.environ["PG_DSN"]) as conn:     # commit on success, rollback on error
        with conn.cursor() as cur:
            # order matters: documents first (the other two tables point to it)
            for b in batches(documents, batch_size):
                cur.executemany(UPSERT_DOCUMENT, b)
            for b in batches(details, batch_size):
                cur.executemany(UPSERT_DETAILS, b)
            for b in batches(chunks, batch_size):
                cur.executemany(UPSERT_CHUNK, b)

            cur.execute(DEACTIVATE_DOCS, (repo, scope_prefixes, doc_ids))
            gone_docs = cur.rowcount
            cur.execute(DEACTIVATE_STALE_CHUNKS, (doc_ids, chunk_ids))
            gone_chunks = cur.rowcount
            cur.execute(DEACTIVATE_ORPHAN_CHUNKS)
            gone_chunks += cur.rowcount

            cur.execute(SUMMARY)
            n_docs, n_details, n_chunks, pending, inactive = cur.fetchone()
    print("\nLoaded into PostgreSQL (active source_code rows):")
    print(f"  documents={n_docs}  source_code_details={n_details}  chunks={n_chunks}")
    print(f"  waiting_for_embedding={pending}   inactive chunks={inactive}")
    if gone_docs or gone_chunks:
        print(f"  deactivated: {gone_docs} documents, {gone_chunks} chunks (no longer in the source)")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Ingest NOVA source code into PostgreSQL")
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT),
                    help=f"source folder (default: {DEFAULT_ROOT})")
    ap.add_argument("--repo", help="repo name stored in source_code_details (default: folder name)")
    ap.add_argument("--only-service", nargs="+", metavar="FOLDER",
                    help="only these top-level folders, e.g. nova-medical nova-web")
    ap.add_argument("--include-infra", action="store_true",
                    help="also ingest CloudFormation / CI / deploy-template yaml")
    ap.add_argument("--commit-sha", default=os.environ.get("COMMIT_SHA"),
                    help="stored in source_code_details.commit_sha (optional)")
    ap.add_argument("--dry-run", action="store_true", help="parse and report only, no database")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-chunk-tokens", type=int, default=DEFAULT_MAX_CHUNK_TOKENS,
                    help=f"maximum tokenizer-counted tokens per chunk, default {DEFAULT_MAX_CHUNK_TOKENS}")
    ap.add_argument("--tokenizer-repo", default=DEFAULT_TOKENIZER_REPO,
                    help=f"Hugging Face repository containing tokenizer.json (default: {DEFAULT_TOKENIZER_REPO})")
    ap.add_argument("--embedding-prefix", default=DEFAULT_EMBEDDING_PREFIX,
                    help=f"prefix sent to the embedding model (default: {DEFAULT_EMBEDDING_PREFIX!r})")
    args = ap.parse_args()

    if args.max_chunk_tokens < MIN_CHUNK_TOKENS:
        sys.exit(f"--max-chunk-tokens must be at least {MIN_CHUNK_TOKENS}")
    if args.max_chunk_tokens > 480:
        sys.exit("--max-chunk-tokens must be 480 or less to leave a safety margin below the model's 512-token context window.")

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    if not args.dry_run and "PG_DSN" not in os.environ:
        sys.exit("Set PG_DSN first, e.g.  $env:PG_DSN = 'postgresql://postgres:<password>@localhost:5432/ragdb'")
    repo = args.repo or root.name

    try:
        tokenizer = Tokenizer.from_file(hf_hub_download(repo_id=args.tokenizer_repo, filename="tokenizer.json"))
    except Exception as exc:
        sys.exit(
            "Could not load tokenizer.json from Hugging Face. Check internet access and "
            "install dependencies with: pip install tokenizers huggingface_hub\n"
            f"Details: {exc}"
        )

    files, sensitive = discover(root, set(args.only_service or []), args.include_infra)
    if not files:
        sys.exit("No source files found.")

    warnings, documents, details, chunks = [], [], [], []
    seen_ids = Counter()
    stats = Counter()
    for f in files:
        if f["path"].stat().st_size > MAX_FILE_BYTES:
            warnings.append(f"{f['rel']}: skipped (larger than {MAX_FILE_BYTES} bytes)")
            continue
        text = f["path"].read_text(encoding="utf-8", errors="replace")
        try:
            d, dt, c = build_records(f, text, repo, args.commit_sha, seen_ids, stats, warnings,
                                     args.max_chunk_tokens, tokenizer, args.embedding_prefix)
        except ValueError as exc:
            warnings.append(f"{f['rel']}: skipped ({exc})")
            continue
        documents += d
        details += dt
        chunks += c
        stats[f"files:{f['layer']}"] += 1

    if not documents:
        sys.exit("Nothing to load.")

    over = sum(1 for c in chunks if c["token_count"] > args.max_chunk_tokens)
    if over:
        sys.exit(f"BUG: {over} chunks exceed the limit of {args.max_chunk_tokens} tokens.")

    parts = Counter(c["doc_id"] for c in chunks)
    split = sum(1 for v in parts.values() if v > 1)
    by_layer = Counter(d["metadata"]["layer"] for d in documents)
    by_kind = Counter(d["metadata"]["symbol_kind"] for d in documents)
    by_service = Counter(d["metadata"]["service"] for d in documents)

    print(f"Read {sum(v for k, v in stats.items() if k.startswith('files:'))} files from {root}")
    print(f"  -> {len(documents)} documents, {len(details)} source_code_details, {len(chunks)} chunks")
    print(f"  split into several chunks: {split} | longest chunk: {max(c['token_count'] for c in chunks)} tokens "
          f"(limit {args.max_chunk_tokens})")
    print(f"  tokenizer: {args.tokenizer_repo} | prefix counted: {args.embedding_prefix!r}")
    print(f"  documents by layer:   {dict(by_layer.most_common())}")
    print(f"  documents by kind:    {dict(by_kind.most_common())}")
    print(f"  documents by service: {dict(by_service.most_common())}")
    print(f"  secret values redacted: {stats['redacted']}")
    print(f"  sensitive files skipped (never read): {len(sensitive)}")
    for s in sensitive[:5]:
        print("    ", s)
    if len(sensitive) > 5:
        print(f"     ... {len(sensitive) - 5} more")
    for w in warnings[:10]:
        print("  WARN", w)
    if len(warnings) > 10:
        print(f"  ... {len(warnings) - 10} more warnings")

    if args.dry_run:
        print("\n--dry-run: nothing written. First document:")
        print(json.dumps({k: documents[0][k] for k in ("doc_id", "title", "content_hash")}, indent=2))
        return

    scope = [f"{s}/" for s in args.only_service] if args.only_service else [""]
    write_to_postgres(documents, details, chunks, repo, scope, args.batch_size)


if __name__ == "__main__":
    main()
