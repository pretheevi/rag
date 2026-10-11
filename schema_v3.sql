CREATE EXTENSION IF NOT EXISTS vector;

-- 1) COMMON HEADER: one row per retrievable unit (a test case, a Jira ticket,
--    a code function, a doc). This is what the LLM receives after a search hit.
CREATE TABLE documents (
    doc_id       TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL CHECK (source_type IN ('testcase', 'jira_ticket', 'source_code', 'doc')),
    title        TEXT,
    content      TEXT NOT NULL,                 -- full text of the unit
    content_hash TEXT NOT NULL,                 -- change detection
    metadata     JSONB NOT NULL DEFAULT '{}',   -- anything without its own column
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, source_type)                -- lets child tables lock the type
);

-- 2) PER-TYPE HEADERS: same doc_id as documents (1 to 1), typed columns
CREATE TABLE testcase_details (
    doc_id      TEXT PRIMARY KEY,
    source_type TEXT NOT NULL DEFAULT 'testcase' CHECK (source_type = 'testcase'),
    file_name   TEXT,
    sheet_name  TEXT,
    module      TEXT NOT NULL,
    section     TEXT,
    tc_label    TEXT,                           -- display only, not identity
    tc_number   TEXT,
    excel_row   INT,
    tc_source   TEXT,
    high_level  BOOLEAN NOT NULL DEFAULT FALSE,
    authority   TEXT NOT NULL DEFAULT 'style_reference',
    preconditions TEXT,
    scenario    TEXT,
    steps       TEXT,
    test_data   TEXT,
    expected    TEXT,
    FOREIGN KEY (doc_id, source_type) REFERENCES documents (doc_id, source_type) ON DELETE CASCADE
);

CREATE TABLE jira_ticket_details (
    doc_id              TEXT PRIMARY KEY,
    source_type         TEXT NOT NULL DEFAULT 'jira_ticket' CHECK (source_type = 'jira_ticket'),
    jira_key            TEXT NOT NULL UNIQUE,   -- e.g. PROJ-123
    project             TEXT,
    issue_type          TEXT,
    status              TEXT,
    priority            TEXT,
    components          TEXT[],
    labels              TEXT[],
    epic_key            TEXT,
    reporter            TEXT,
    assignee            TEXT,
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ,
    acceptance_criteria TEXT,
    FOREIGN KEY (doc_id, source_type) REFERENCES documents (doc_id, source_type) ON DELETE CASCADE
);

CREATE TABLE source_code_details (
    doc_id      TEXT PRIMARY KEY,
    source_type TEXT NOT NULL DEFAULT 'source_code' CHECK (source_type = 'source_code'),
    repo        TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    language    TEXT,
    symbol_name TEXT,                           -- function / method / class
    symbol_kind TEXT,
    signature   TEXT,
    start_line  INT,
    end_line    INT,
    commit_sha  TEXT,
    FOREIGN KEY (doc_id, source_type) REFERENCES documents (doc_id, source_type) ON DELETE CASCADE
);

-- 3) CHUNKS: only the embedded pieces, shared by every source type
CREATE TABLE chunks (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    source_type     TEXT NOT NULL,              -- copy of documents.source_type, for fast filtering
    chunk_index     INT  NOT NULL,              -- order inside the document
    content         TEXT NOT NULL,              -- exactly what is embedded (model prefix added at embed time)
    content_hash    TEXT NOT NULL,
    chunk_strategy  TEXT,
    token_count     INT,
    metadata        JSONB NOT NULL DEFAULT '{}',-- part_index, part_count, part_type (description/comment) ...
    embedding_model TEXT,
    embedding       VECTOR(768),                -- must match the embedding model
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    FOREIGN KEY (doc_id, source_type) REFERENCES documents (doc_id, source_type) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_doc      ON chunks (doc_id, chunk_index);
CREATE INDEX idx_chunks_source   ON chunks (source_type) WHERE is_active;
CREATE INDEX idx_chunks_pending  ON chunks (id) WHERE embedding IS NULL AND is_active;
CREATE INDEX idx_chunks_meta     ON chunks USING gin (metadata);
CREATE INDEX idx_tc_module       ON testcase_details (module, high_level);
CREATE INDEX idx_jira_project    ON jira_ticket_details (project, status);
CREATE INDEX idx_code_file       ON source_code_details (repo, file_path);
-- create the vector index AFTER embeddings are filled:
-- CREATE INDEX idx_chunks_vec ON chunks USING hnsw (embedding vector_cosine_ops);