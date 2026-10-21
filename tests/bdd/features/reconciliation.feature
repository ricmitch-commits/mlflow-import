Feature: Reconciliation of previously imported traces
  Re-running the importer recognizes traces it owns, skips unchanged turns,
  and repairs or replaces changed ones without touching foreign traces.
  The reconciliation-seam scenarios pin the plan/apply contract directly;
  the boundary scenarios drive the full CLI pipeline and assert the report,
  the gateway event history, and the stored traces.

  Scenario: Replace changed content without deleting the old trace first
    Given an importer-owned trace exists for session "ses-1" turn "turn-1"
    And the source fingerprint has changed
    When I import the session
    Then the event order is create, verify, upload, verify, retire-old
    And an unowned trace with the same native session remains unchanged

  Scenario: Preserve the old trace when replacement upload fails
    Given a changed turn whose replacement cannot verify its artifact
    When I import the changed turn
    Then the old trace remains active
    And the new trace is rolled back or reported as a recovery conflict

  Scenario: A new experiment name is created once and reused unchanged
    Given a source session with one synthetic turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records the experiment as created
    And the gateway recorded exactly one experiment creation
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records the experiment as resolved with the same experiment id
    And the gateway recorded no new experiment creation
    And the reused experiment kept its original settings

  Scenario: An unchanged rerun skips every owned trace without new mutations
    Given a source session with two synthetic turns
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records both turns as imported
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report marks every turn as skipped unchanged
    And the gateway recorded no new mutations
    And the owned traces were not duplicated

  Scenario: A changed turn is replaced in the safe mutation order
    Given a source session with one synthetic turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    Given the source turn's reply changed
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report marks the turn as updated
    And the report records one retired trace
    And the event order is create, verify, upload, verify, retire-old
    And the old trace is retired and the replacement is active

  Scenario: A failed replacement upload preserves the old trace
    Given a source session with one synthetic turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    Given the source turn's reply changed
    And replacement artifact verification fails
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 1
    And the old trace remains active
    And the failed replacement was rolled back
    And the report marks the turn as failed

  Scenario: A stale turn of the selected session is retired
    Given a source session with two synthetic turns
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    Given the source lost its second turn
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report marks the first turn as skipped unchanged
    And the report marks the lost turn as a retired stale turn
    And the gateway no longer stores the retired trace

  Scenario: Retirement is scoped to the selected session
    Given a synthetic session "ses-bdd-scope-alpha" created at "2026-10-21T12:00:00Z" in "/repo/bdd-alpha" with 1 turn
    And a synthetic session "ses-bdd-scope-beta" created at "2026-10-21T12:00:00Z" in "/repo/bdd-beta" with 1 turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    Given the source lost every turn of session "ses-bdd-scope-alpha"
    When I rerun the import through the CLI with fake dependencies filtered to directory "/repo/bdd-beta"
    Then the process exits with status 0
    And the report counts one unexamined owned trace
    And the owned traces of session "ses-bdd-scope-alpha" remain stored
    When I rerun the import through the CLI with fake dependencies without filters
    Then the process exits with status 0
    And the report marks every trace of session "ses-bdd-scope-alpha" as retired tombstones
    And the gateway no longer stores any trace of session "ses-bdd-scope-alpha"

  Scenario: A foreign trace sharing the native session id is never touched
    Given a source session with one synthetic turn
    And an unowned foreign trace sharing the native session id exists in the experiment
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records 1 imported turn
    And the foreign trace remains stored and unchanged
    And no gateway event targeted the foreign trace
