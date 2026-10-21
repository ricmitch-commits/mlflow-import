Feature: Read-only source snapshot safety
  The importer never mutates the OpenCode SQLite database it reads: it opens
  a consistent read-only snapshot, extracts only permitted records, and keeps
  the live source connection closed for the rest of the run. Earlier
  scenarios reconstruct turns through the public source pipeline — fixture
  database, normalized snapshot, replay adapter, logical turns — while the
  boundary scenarios drive the full CLI pipeline over a live WAL-mode
  source and an unsupported schema.

  Scenario: Consecutive user prompts close the previous turn
    Given a current source session "ses-bdd-current-001" with consecutive user prompts "bdd-user-1" and "bdd-user-2"
    When the source pipeline reconstructs the session's logical turns
    Then the session has 2 logical turns
    And turn 1 has terminal outcome "no_assistant_result"
    And turn 2 starts at user prompt "bdd-user-2"

  Scenario: Legacy parts follow their parent message
    Given a legacy source session "ses-bdd-legacy-001" with a user message "bdd-user-1" followed by parts "bdd-part-1" and "bdd-part-2"
    When the source pipeline reconstructs the session's logical turns
    Then the session has 1 logical turn
    And turn 1 replays the records "bdd-user-1, bdd-part-1, bdd-part-2"

  Scenario: Two identical missing-ID records stay distinct
    Given a legacy source session "ses-bdd-legacy-002" with 2 identical missing-ID user messages
    When the source pipeline reconstructs the session's logical turns
    Then the session has 2 logical turns
    And the turns have distinct turn keys
    And turn 1 records the warning "turn_key_fallback_identity"

  Scenario: Records before the first user prompt stay unassigned
    Given a current source session "ses-bdd-current-002" with a system record before user prompt "bdd-user-3"
    When the source pipeline reconstructs the session's logical turns
    Then the session has 1 logical turn
    And turn 1 replays the records "bdd-user-3, bdd-user-3#content-0"

  Scenario: A live WAL-mode source is read without any source write
    Given a live WAL-mode source database with one synthetic turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records 1 imported turn for session "ses-bdd-wal"
    And the source database bytes are unchanged
    And the source WAL companion is unchanged
    And the source SHM companion keeps its size and inode

  Scenario: An unsupported schema exits with status two and a safe diagnostic
    Given a source database with an unsupported schema
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 2
    And the report carries the safe error "unsupported_schema"
    And the fake gateway recorded no mutations
    And the fake lock factory recorded no acquisitions
