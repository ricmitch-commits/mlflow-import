Feature: CLI invocation and session selection
  The importer runs as a one-shot command that opens a read-only OpenCode
  snapshot, selects exactly the sessions the operator requested, and imports
  each selected logical turn as one importer-owned MLflow trace.

  Scenario: Reject a timestamp without an explicit timezone
    When I invoke the importer with "--from-time 2026-10-21T12:00:00"
    Then the process exits with status 2
    And the safe error code is "invalid_timestamp"
    And the result does not contain "2026-10-21T12:00:00"

  Scenario: Normalize an incomplete model identity exactly
    Given a session contains the model identity "provider/"
    When I select model " provider/ "
    Then the session is selected
    And a complete model request does not match "provider/"

  Scenario: Include exact time boundaries and descendants
    Given a session was created at "2026-10-21T12:00:00Z" in "/repo/project/app"
    When I filter from "2026-10-21T12:00:00Z" to "2026-10-21T12:00:00Z" and directory "/repo/project"
    Then the session is selected
    And a session in "/repo/projector" is not selected

  Scenario: Apply a strict turn threshold
    Given a session has 2 logical user turns
    When I select sessions with minimum user turns 2
    Then the session is not selected
    When I select sessions with minimum user turns 1
    Then the session is selected with all 2 turns

  Scenario: Exact time bounds are inclusive and apply to creation time only
    Given a synthetic session "ses-bdd-time-at-from" created at "2026-10-21T12:00:00Z" in "/repo/bdd-time" with 1 turn
    And a synthetic session "ses-bdd-time-at-to" created at "2026-10-21T13:00:00Z" in "/repo/bdd-time" with 1 turn
    And a synthetic session "ses-bdd-time-early" created at "2026-10-21T11:00:00Z" in "/repo/bdd-time" with message activity at "2026-10-21T12:30:00Z" and 1 turn
    And a synthetic session "ses-bdd-time-late" created at "2026-10-21T14:00:00Z" in "/repo/bdd-time" with 1 turn
    When I run an import through the CLI with fake dependencies from "2026-10-21T12:00:00Z" to "2026-10-21T13:00:00Z"
    Then the process exits with status 0
    And only the sessions "ses-bdd-time-at-from, ses-bdd-time-at-to" were selected
    And the report records 2 imported turns

  Scenario: The directory filter selects descendants and excludes sibling prefixes
    Given a synthetic session "ses-bdd-dir-child" created at "2026-10-21T12:00:00Z" in "/repo/bdd-dir/project/app" with 1 turn
    And a synthetic session "ses-bdd-dir-root" created at "2026-10-21T12:00:00Z" in "/repo/bdd-dir/project" with 1 turn
    And a synthetic session "ses-bdd-dir-sibling" created at "2026-10-21T12:00:00Z" in "/repo/bdd-dir/projector" with 1 turn
    When I run an import through the CLI with fake dependencies in directory "/repo/bdd-dir/project"
    Then the process exits with status 0
    And only the sessions "ses-bdd-dir-child, ses-bdd-dir-root" were selected
    And the report records 2 imported turns

  Scenario: All four selection filters combine at the CLI boundary
    Given a synthetic session "ses-bdd-and-target" created at "2026-10-21T12:00:00Z" in "/repo/bdd-and" with 4 turns and message model "prov-and/model-one"
    And a synthetic session "ses-bdd-and-time" created at "2026-10-21T11:00:00Z" in "/repo/bdd-and" with 4 turns and message model "prov-and/model-one"
    And a synthetic session "ses-bdd-and-directory" created at "2026-10-21T12:00:00Z" in "/repo/bdd-anger" with 4 turns and message model "prov-and/model-one"
    And a synthetic session "ses-bdd-and-model" created at "2026-10-21T12:00:00Z" in "/repo/bdd-and" with 4 turns and message model "prov-and/model-two"
    And a synthetic session "ses-bdd-and-turns" created at "2026-10-21T12:00:00Z" in "/repo/bdd-and" with 3 turns and message model "prov-and/model-one"
    When I run an import through the CLI with fake dependencies from "2026-10-21T12:00:00Z" to "2026-10-21T14:00:00Z" in directory "/repo/bdd-and" for model "prov-and/model-one" with minimum user turns 3
    Then the process exits with status 0
    And only the sessions "ses-bdd-and-target" were selected
    And the report records 4 imported turns for session "ses-bdd-and-target"

  Scenario: The model filter matches identities recorded on messages and parts
    Given a synthetic session "ses-bdd-model-message" created at "2026-10-21T12:00:00Z" in "/repo/bdd-model" with 1 turn and message model "prov-msg/model-one"
    And a synthetic session "ses-bdd-model-part" created at "2026-10-21T12:00:00Z" in "/repo/bdd-model" with 1 turn and part model "prov-part/model-two"
    When I run an import through the CLI with fake dependencies for model "prov-msg/model-one"
    Then the process exits with status 0
    And only the sessions "ses-bdd-model-message" were selected
    When I run an import through the CLI with fake dependencies for model "prov-part/model-two"
    Then the process exits with status 0
    And only the sessions "ses-bdd-model-part" were selected

  Scenario: A legacy source imports through the CLI with a message-level model filter
    Given a synthetic legacy session "ses-bdd-legacy-model" created at "2026-10-21T12:00:00Z" with 1 turn and message model "prov-legacy/model-old"
    When I run an import through the CLI with fake dependencies for model "prov-legacy/model-old"
    Then the process exits with status 0
    And only the sessions "ses-bdd-legacy-model" were selected
    And the report records 1 imported turn for session "ses-bdd-legacy-model"
