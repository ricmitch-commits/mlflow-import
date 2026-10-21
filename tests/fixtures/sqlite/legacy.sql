-- Sanitized synthetic LEGACY-generation OpenCode fixture.
-- Every ID, prompt, path, and credential-looking value is synthetic; no
-- real session data appears here. The catalogued legacy tables (session,
-- message, part, and the session-scoped todo table) carry the test rows.
-- The prohibited secret tables are present with sentinel values so tests
-- can assert their rows never reach a normalized snapshot.

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

INSERT INTO session VALUES (
    'ses-legacy-001', 'proj-synth-2', 'ws-synth-2', NULL,
    'synthetic-legacy-one', '/repo/project-beta', '/repo/project-beta/work',
     'Synthetic legacy session one', 'synthetic-legacy-version', 'SHARE_URL_FIXTURE_SENTINEL_LEGACY',
    2, 1, 1, '{"modified": ["synthetic-legacy-file"]}',
    '{"environment": "synthetic-legacy"}', 0.5,
    80, 210, 5, 40, 2,
    NULL, NULL, 'synthetic-legacy-agent', '{"provider": "synth-provider", "model": "synth-legacy-model"}',
    1771000000000, 1771000800000, NULL, NULL, 'synthetic-legacy-flag'
);

INSERT INTO session VALUES (
    'ses-legacy-002', 'proj-synth-2', NULL, 'ses-legacy-001',
    'synthetic-legacy-two', '/repo/project-beta/nested', NULL,
    'Synthetic legacy session two', 'synthetic-legacy-version', NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    1771000500000, 1771000600000, 1771000700000, 1771000900000, NULL
);

-- A session without a creation timestamp: it must sort after every
-- timestamped session (by session ID) without being dropped.
INSERT INTO session VALUES (
    'ses-legacy-003', NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL
);

INSERT INTO message VALUES (
    'msg-legacy-001', 'ses-legacy-001', 1771000010000, 1771000010000,
    '{"role": "user", "content": "synthetic legacy user prompt"}'
);

INSERT INTO message VALUES (
    'msg-legacy-002', 'ses-legacy-001', 1771000020000, 1771000025000,
    '{"role": "assistant", "model": {"provider": "synth-provider", "model": "synth-legacy-model"}}'
);

INSERT INTO message VALUES (
    'msg-legacy-101', 'ses-legacy-002', 1771000510000, 1771000510000,
    '{"role": "user", "content": "synthetic legacy child prompt"}'
);

INSERT INTO message VALUES (
    'msg-legacy-301', 'ses-legacy-003', NULL, NULL,
    '{"role": "user", "content": "synthetic legacy untimed prompt"}'
);

INSERT INTO part VALUES (
    'part-legacy-001', 'msg-legacy-001', 'ses-legacy-001', 1771000011000, 1771000011000,
    '{"type": "text", "text": "synthetic legacy user text"}'
);

INSERT INTO part VALUES (
    'part-legacy-002', 'msg-legacy-001', 'ses-legacy-001', 1771000012000, 1771000012000,
    '{"type": "tool", "tool": "synthetic-legacy-tool", "state": {"output": "synthetic tool output"}}'
);

INSERT INTO part VALUES (
    'part-legacy-003', 'msg-legacy-002', 'ses-legacy-001', 1771000021000, 1771000021000,
    '{"type": "step-start"}'
);

INSERT INTO part VALUES (
    'part-legacy-004', 'msg-legacy-002', 'ses-legacy-001', 1771000022000, 1771000022000,
    '{"type": "text", "text": "synthetic legacy assistant text"}'
);

-- A synthetic step-finish part carrying the model identity at part level,
-- the way legacy OpenCode persists per-step model/usage metadata.
INSERT INTO part VALUES (
    'part-legacy-005', 'msg-legacy-002', 'ses-legacy-001', 1771000023000, 1771000023000,
    '{"type": "step-finish", "model": {"provider": "synth-provider", "model": "synth-legacy-model"}, "usage": {"input": 80, "output": 210}}'
);

-- Malformed JSON data: the raw text is retained, the decoded value stays
-- absent, and the record warns.
INSERT INTO part VALUES (
    'part-legacy-006', 'msg-legacy-001', 'ses-legacy-001', 1771000013000, 1771000013000,
    '{"broken'
);

INSERT INTO todo VALUES (
    'ses-legacy-001', 'synthetic legacy todo content', 'pending', 'high', 1,
    1771000005000, 1771000006000
);

-- An unassociated todo: its session_id matches no session row, so it must
-- be reported (warning) and never assigned to a session.
INSERT INTO todo VALUES (
    'ses-legacy-unknown', 'synthetic orphan todo content', 'pending', 'low', 9,
    1771000001000, 1771000002000
);

INSERT INTO account VALUES ('acct-synth-002', 'synthetic-legacy-account', 'SECRET-LEGACY-ACCOUNT-SENTINEL');
INSERT INTO account_state VALUES ('acct-synth-002', 'SECRET-LEGACY-STATE-SENTINEL');
INSERT INTO control_account VALUES ('ctl-synth-002', 'SECRET-LEGACY-CONTROL-SENTINEL');
INSERT INTO credential VALUES ('cred-synth-002', 'legacy-credential-sentinel');
INSERT INTO session_share VALUES ('share-synth-002', 'ses-legacy-001', 'https://share.example.invalid/legacy-synth', 'SECRET-LEGACY-SHARE-SENTINEL');
