-- Sanitized synthetic CURRENT-generation OpenCode fixture.
-- Every ID, prompt, path, and credential-looking value is synthetic; no
-- real session data appears here. The catalogued current tables
-- (session_v2, session_message, and the session-scoped tables) carry the
-- test rows. The prohibited secret tables (account, account_state,
-- control_account, credential, session_share) are present with sentinel
-- values so tests can assert their rows never reach a normalized
-- snapshot.

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
    resume_attempts INTEGER,
    future_note TEXT
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

INSERT INTO session_v2 VALUES (
    'ses-current-001', 'proj-synth-1', 'ws-synth-1', NULL, NULL, NULL,
    'synthetic-current-one', '/repo/project-alpha', '/repo/project-alpha/work',
     'Synthetic current session one', 'synthetic-version', 'SHARE_URL_FIXTURE_SENTINEL_CURRENT',
    3, 1, 2, '{"modified": ["synthetic-file-a", "synthetic-file-b"]}',
    '{"environment": "synthetic"}', 0.125,
    120, 340, 12, 90, 8,
    NULL, NULL, 'synthetic-agent', '{"provider": "synth-provider", "model": "synth-model"}',
    1772000000000, 1772000900000, NULL, 1772000800000, NULL,
    NULL, NULL, NULL, NULL, 'synthetic-future-note'
);

INSERT INTO session_v2 VALUES (
    'ses-current-002', 'proj-synth-1', NULL, NULL, 'ses-current-001', 1772000500000,
    'synthetic-current-two', '/repo/project-alpha/nested', NULL,
    NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1772000500000, 1772000600000, 1772000550000, NULL, 'idle-timeout',
    NULL, NULL, NULL, 2, NULL
);

INSERT INTO session_message VALUES (
    'msg-current-001', 'ses-current-001', 'user', 1, 1772000010000, 1772000010000,
    '{"content": [{"type": "text", "text": "synthetic user prompt one"}], "model": {"provider": "synth-provider", "model": "synth-model"}}'
);

INSERT INTO session_message VALUES (
    'msg-current-002', 'ses-current-001', 'assistant', 2, 1772000020000, 1772000025000,
    '{"content": [{"type": "text", "text": "synthetic assistant reply one"}], "model": {"provider": "synth-provider", "model": "synth-model"}}'
);

INSERT INTO session_message VALUES (
    'msg-current-003', 'ses-current-001', 'user', 3, 1772000030000, 1772000030000,
    '{"content": [{"type": "text", "text": "synthetic user prompt two"}]}'
);

INSERT INTO session_message VALUES (
    'msg-current-004', 'ses-current-001', 'system', 4, 1772000040000, 1772000040000,
    '{"synthetic": true, "content": [{"type": "text", "text": "synthetic system notice"}]}'
);

INSERT INTO session_message VALUES (
    'msg-current-101', 'ses-current-002', 'user', 1, 1772000510000, 1772000510000,
    '{"content": [{"type": "text", "text": "synthetic fork user prompt"}]}'
);

INSERT INTO session_pending VALUES (
    'pending-001', 'ses-current-001', 'tool', '{"name": "synthetic-tool"}',
    'polled', 2, 1772000022000
);

INSERT INTO session_inbox VALUES (
    'inbox-001', 'ses-current-001', 'event', '{"event": "synthetic-event"}',
    'queued', 1, 1772000012000
);

INSERT INTO instruction_entry VALUES (
    'ses-current-001', 'synthetic-instruction-key', 'synth-blob-hash-001',
    0, 1772000005000, 1772000006000
);

INSERT INTO instruction_blob VALUES (
    'synth-blob-hash-001', 'synthetic instruction blob content'
);

INSERT INTO instruction_state VALUES (
    'ses-current-001', 1, 4,
    '{"synthetic-instruction-key": "synth-blob-hash-001"}',
    '{"synthetic-instruction-key": "synth-blob-hash-001"}'
);

-- A second session's instruction entry referencing the SAME blob hash as
-- ses-current-001's entry: the blob must be emitted exactly once (explicit
-- hash identity), under the first session that references it.
INSERT INTO instruction_entry VALUES (
    'ses-current-002', 'synthetic-instruction-key-two', 'synth-blob-hash-001',
    0, 1772000610000, 1772000610000
);

-- Unassociated session-scoped rows: their session_id values match no
-- session_v2 row, so they must be reported (warnings) and never assigned.
INSERT INTO session_pending VALUES (
    'pending-orphan-001', 'ses-current-unknown', 'tool', '{"name": "synthetic-orphan-tool"}',
    'polled', 1, 1772000003000
);

INSERT INTO instruction_entry VALUES (
    'ses-current-unknown', 'synthetic-orphan-instruction-key', 'synth-blob-hash-orphan',
    0, 1772000004000, 1772000005000
);

-- An instruction blob referenced by no associated instruction entry.
INSERT INTO instruction_blob VALUES (
    'synth-blob-hash-orphan', 'synthetic orphan instruction blob content'
);

INSERT INTO account VALUES ('acct-synth-001', 'synthetic-account', 'SECRET-ACCOUNT-TOKEN-SENTINEL');
INSERT INTO account_state VALUES ('acct-synth-001', 'SECRET-ACCOUNT-STATE-SENTINEL');
INSERT INTO control_account VALUES ('ctl-synth-001', 'SECRET-CONTROL-ACCOUNT-SENTINEL');
INSERT INTO credential VALUES ('cred-synth-001', 'credential-sentinel');
INSERT INTO session_share VALUES ('share-synth-001', 'ses-current-001', 'https://share.example.invalid/synth', 'SECRET-SHARE-SENTINEL');
