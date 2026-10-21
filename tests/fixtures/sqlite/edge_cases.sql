-- Sanitized synthetic EDGE-CASE fixture for the current generation.
--
-- Malformed/unsupported schema: a legacy `message` table exists but is
-- MISSING the required `data` column, so the legacy catalog entry must
-- NOT match (the database classifies as "current", never "mixed").
--
-- Optional/unknown session columns: `session_v2` omits the optional
-- `time_suspended` and `resume_attempts` columns and adds an unknown
-- additive column `custom_flag`; the current catalog entry still matches
-- and the unknown column is simply observed.
--
-- Edge values in `session_message`: NULL id, NULL seq, NULL time_created,
-- NULL data, duplicate-key JSON text, a BLOB holding valid UTF-8 JSON,
-- and a BLOB holding invalid UTF-8 bytes (SQLite cannot return invalid
-- UTF-8 through a TEXT column, so binary payloads arrive as BLOBs).
--
-- `session_share` is present so schema probing reports the omission
-- warning; no other secret tables are included.

CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    fork_session_id TEXT,
    fork_boundary INTEGER,
    slug TEXT,
    directory TEXT,
    path TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    metadata TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_reasoning INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_write INTEGER,
    revert TEXT,
    permission TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_idle INTEGER,
    time_viewed INTEGER,
    idle_outcome TEXT,
    time_compacting INTEGER,
    time_archived INTEGER,
    custom_flag TEXT
);

CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    seq INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);

-- Malformed legacy generation: `message` is missing the required `data` column.
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    time_created INTEGER,
    time_updated INTEGER
);

CREATE TABLE session_share (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    share_url TEXT,
    secret TEXT
);

-- Session with missing (NULL) values for several optional fields and an
-- unknown additive column value.
INSERT INTO session_v2 (id, slug, custom_flag)
VALUES ('ses-edge-001', 'synthetic-edge-one', 'unknown-column-value');

INSERT INTO session_message VALUES (
    'msg-edge-001', 'ses-edge-001', 'user', 1, 1774000010000, 1774000010000,
    '{"content": [{"type": "text", "text": "synthetic edge prompt"}]}'
);

-- Duplicate JSON object keys in the raw text.
INSERT INTO session_message VALUES (
    'msg-edge-002', 'ses-edge-001', 'assistant', 2, 1774000020000, 1774000020000,
    '{"text": "first", "text": "second"}'
);

-- BLOB column value holding valid UTF-8 JSON bytes.
INSERT INTO session_message VALUES (
    'msg-edge-003', 'ses-edge-001', 'assistant', 3, 1774000030000, 1774000030000,
    CAST('{"content": [{"type": "text", "text": "synthetic blob json"}]}' AS BLOB)
);

-- BLOB column value holding invalid UTF-8 bytes.
INSERT INTO session_message VALUES (
    'msg-edge-004', 'ses-edge-001', 'assistant', 4, 1774000040000, 1774000040000,
    CAST(x'fffe80' AS BLOB)
);

-- NULL data, NULL seq, NULL time_created, and a NULL (missing) row id.
INSERT INTO session_message VALUES (
    'msg-edge-005', 'ses-edge-001', 'user', NULL, NULL, NULL, NULL
);

INSERT INTO session_message VALUES (
    NULL, 'ses-edge-001', 'assistant', 6, 1774000060000, 1774000060000,
    '{"content": [{"type": "text", "text": "synthetic edge reply without id"}]}'
);

INSERT INTO session_share VALUES ('share-synth-003', 'ses-edge-001', 'https://share.example.invalid/edge-synth', 'SECRET-EDGE-SHARE-SENTINEL');
