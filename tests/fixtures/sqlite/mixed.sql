-- Sanitized synthetic MIXED-generation OpenCode fixture.
-- Both catalogued generations exist in one database: the legacy tables
-- (session/message/part/todo) and the current tables (session_v2/
-- session_message plus its session-scoped tables). Session
-- 'ses-mixed-shared-001' exists in BOTH session tables so mixed-mode
-- precedence (current authoritative, legacy suppressed) is exercised by
-- the adapter tasks. The prohibited secret tables are present with
-- sentinel values so adapter tests can assert their rows never reach a
-- normalized snapshot; the dedicated current/legacy fixtures carry the
-- same sentinel exclusion rows.

CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
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
    time_compacting INTEGER,
    time_archived INTEGER,
    custom_flag TEXT
);

CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);

CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);

CREATE TABLE todo (
    session_id TEXT NOT NULL,
    content TEXT,
    status TEXT,
    priority TEXT,
    position INTEGER,
    time_created INTEGER,
    time_updated INTEGER
);

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
    time_suspended INTEGER,
    resume_attempts INTEGER
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

CREATE TABLE session_pending (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT,
    data TEXT,
    delivery TEXT,
    admitted_seq INTEGER,
    time_created INTEGER
);

CREATE TABLE session_inbox (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT,
    payload TEXT,
    delivery TEXT,
    enqueued_seq INTEGER,
    time_created INTEGER
);

CREATE TABLE instruction_entry (
    session_id TEXT NOT NULL,
    key TEXT,
    value TEXT,
    removed INTEGER,
    time_created INTEGER,
    time_updated INTEGER
);

CREATE TABLE instruction_blob (
    hash TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE instruction_state (
    session_id TEXT NOT NULL,
    epoch_start INTEGER,
    through_seq INTEGER,
    initial_values TEXT,
    current_values TEXT
);

-- Legacy-only session (no session_v2 row with this ID).
INSERT INTO session VALUES (
    'ses-mixed-legacy-001', 'proj-synth-3', NULL, NULL,
    'synthetic-mixed-legacy', '/repo/project-mixed/legacy', NULL,
    'Synthetic mixed legacy session', 'synthetic-version', NULL,
    1, 0, 1, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1773000000000, 1773000100000, NULL, NULL, 'synthetic-mixed-flag'
);

INSERT INTO message VALUES (
    'msg-mixed-legacy-001', 'ses-mixed-legacy-001', 1773000010000, 1773000010000,
    '{"role": "user", "content": "synthetic mixed legacy prompt"}'
);

INSERT INTO part VALUES (
    'part-mixed-legacy-001', 'msg-mixed-legacy-001', 'ses-mixed-legacy-001',
    1773000011000, 1773000011000,
    '{"type": "text", "text": "synthetic mixed legacy part"}'
);

-- Malformed JSON part data: raw text retained, decoded value absent.
INSERT INTO part VALUES (
    'part-mixed-legacy-002', 'msg-mixed-legacy-001', 'ses-mixed-legacy-001',
    1773000011500, 1773000011500,
    '{"broken'
);

-- A synthetic step-finish part carrying the model identity at part level.
INSERT INTO part VALUES (
    'part-mixed-legacy-003', 'msg-mixed-legacy-001', 'ses-mixed-legacy-001',
    1773000012000, 1773000012000,
    '{"type": "step-finish", "model": {"provider": "synth-provider", "model": "synth-mixed-model"}}'
);

-- An unassociated todo: reported, never assigned.
INSERT INTO todo VALUES (
    'ses-mixed-unknown', 'synthetic mixed orphan todo', 'pending', 'low', 3,
    1773000001000, 1773000001000
);

-- Current sessions. ses-mixed-current-002 is created BETWEEN the legacy
-- session and the shared session so mixed export order interleaves the
-- generations by normalized creation time.
INSERT INTO session_v2 VALUES (
    'ses-mixed-current-001', 'proj-synth-3', NULL, NULL, NULL, NULL,
    'synthetic-mixed-current', '/repo/project-mixed/current', NULL,
    'Synthetic mixed current session', 'synthetic-version', NULL,
    1, 1, 1, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1773000500000, 1773000600000, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL
);

INSERT INTO session_message VALUES (
    'msg-mixed-current-001', 'ses-mixed-current-001', 'user', 1, 1773000510000, 1773000510000,
    '{"content": [{"type": "text", "text": "synthetic mixed current prompt"}, {"type": "text", "text": "synthetic mixed current follow-up"}], "model": {"provider": "synth-provider", "model": "synth-mixed-model"}}'
);

INSERT INTO session_v2 VALUES (
    'ses-mixed-current-002', 'proj-synth-3', NULL, NULL, NULL, NULL,
    'synthetic-mixed-early-current', '/repo/project-mixed/early', NULL,
    'Synthetic mixed early current session', 'synthetic-version', NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1773000050000, 1773000050000, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL
);

INSERT INTO session_message VALUES (
    'msg-mixed-current-002', 'ses-mixed-current-002', 'assistant', 1, 1773000051000, 1773000051000,
    '{"content": [{"type": "text", "text": "synthetic mixed early current reply"}], "model": {"provider": "synth-provider", "model": "synth-mixed-model"}}'
);

-- Overlapping session: same ID in both generations; current is
-- authoritative for mixed databases.
INSERT INTO session VALUES (
    'ses-mixed-shared-001', 'proj-synth-3', NULL, NULL,
    'synthetic-mixed-shared', '/repo/project-mixed/shared', NULL,
    'Synthetic mixed shared (legacy row)', 'synthetic-version', NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1773000200000, 1773000200000, NULL, NULL, NULL
);

INSERT INTO message VALUES (
    'msg-mixed-shared-legacy-001', 'ses-mixed-shared-001', 1773000210000, 1773000210000,
    '{"role": "user", "content": "synthetic mixed shared legacy prompt"}'
);

INSERT INTO session_v2 VALUES (
    'ses-mixed-shared-001', 'proj-synth-3', NULL, NULL, NULL, NULL,
    'synthetic-mixed-shared', '/repo/project-mixed/shared', NULL,
    'Synthetic mixed shared (current row)', 'synthetic-version', NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1773000250000, 1773000250000, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL
);

INSERT INTO session_message VALUES (
    'msg-mixed-shared-current-001', 'ses-mixed-shared-001', 'user', 1, 1773000260000, 1773000260000,
    '{"content": [{"type": "text", "text": "synthetic mixed shared current prompt"}]}'
);

INSERT INTO session_pending VALUES (
    'pending-mixed-001', 'ses-mixed-current-001', 'tool', '{"name": "synthetic-mixed-tool"}',
    'polled', 1, 1773000512000
);

-- Unassociated current session-scoped rows: reported, never assigned.
INSERT INTO session_pending VALUES (
    'pending-mixed-orphan', 'ses-mixed-unknown', 'tool', '{"name": "synthetic-mixed-orphan-tool"}',
    'polled', 1, 1773000002000
);

INSERT INTO instruction_entry VALUES (
    'ses-mixed-unknown', 'synthetic-mixed-orphan-key', 'synth-mixed-orphan-hash',
    0, 1773000003000, 1773000003000
);

-- Prohibited secret tables with sentinel values: their rows must never
-- reach a normalized snapshot.
CREATE TABLE account (
    id TEXT PRIMARY KEY,
    name TEXT,
    token TEXT
);

CREATE TABLE account_state (
    account_id TEXT PRIMARY KEY,
    state TEXT
);

CREATE TABLE control_account (
    id TEXT PRIMARY KEY,
    config TEXT
);

CREATE TABLE credential (
    id TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE session_share (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    share_url TEXT,
    secret TEXT
);

INSERT INTO account VALUES ('acct-synth-mixed-001', 'synthetic-mixed-account', 'SECRET-MIXED-ACCOUNT-TOKEN-SENTINEL');
INSERT INTO account_state VALUES ('acct-synth-mixed-001', 'SECRET-MIXED-ACCOUNT-STATE-SENTINEL');
INSERT INTO control_account VALUES ('ctl-synth-mixed-001', 'SECRET-MIXED-CONTROL-ACCOUNT-SENTINEL');
INSERT INTO credential VALUES ('cred-synth-mixed-001', 'mixed-credential-sentinel');
INSERT INTO session_share VALUES ('share-synth-mixed-001', 'ses-mixed-shared-001', 'https://share.example.invalid/mixed-synth', 'SECRET-MIXED-SHARE-SENTINEL');
