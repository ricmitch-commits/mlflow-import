Feature: Privacy, redaction, and reporting
  Redaction mode deterministically protects identifiers before anything is
  persisted, and every run ends with a safe human or machine-readable report.
  The redaction-seam scenario pins the pure redactor; the boundary scenarios
  drive the full CLI pipeline and assert what was persisted and reported.

  Scenario: Redact trace content and artifact content without changing identity
    Given a turn contains an authorization bearer token and source ID "part-7"
    When I import with redaction enabled
    Then the trace and artifact contain "[REDACTED]"
    And neither contains the original bearer token
    And the source ID remains "part-7"
    And the report contains detector counts but not the original token

  Scenario: A dry-run reports planned actions without mutating MLflow
    Given a source session with one synthetic turn
    When I run a dry-run import through the CLI with fake dependencies
    Then the process exits with status 0
    And the JSON report marks the turn as skipped for dry-run
    And the fake gateway recorded no mutations
    And the fake lock factory recorded no acquisitions

  Scenario: The report never contains source content or secrets
    Given a source session whose prompt contains a synthetic bearer token
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report output contains no prompt content
    And the report output contains no bearer token
    And the report output contains no absolute source path

  Scenario: A capability failure exits with status two and a safe error
    Given a source session with one synthetic turn
    And a tracking target that fails the capability check
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 2
    And the safe error code is "mlflow_capability_unsupported"

  Scenario: An item failure is isolated and exits with status one
    Given a source session with one failing turn write
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 1
    And the JSON report records the failed turn

  Scenario: Redaction masks trace and artifact content through the CLI
    Given a source session whose prompt contains a synthetic bearer token
    When I run a redacting import through the CLI with fake dependencies
    Then the process exits with status 0
    And the stored trace content is masked without the bearer token
    And the stored artifact is masked without the bearer token
    And the report shows redaction counts without the bearer token
    And the redacted identifiers survive

  Scenario: A failing session does not stop later sessions
    Given a synthetic session "ses-bdd-cont-first" created at "2026-10-21T12:00:00Z" with 1 turn and no directory
    And a synthetic session "ses-bdd-cont-second" created at "2026-10-21T12:01:00Z" with 1 turn and no directory
    And the first session's trace write fails
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 1
    And the report records session "ses-bdd-cont-first" as failed
    And the report records session "ses-bdd-cont-second" as imported

  Scenario: The exit status follows the run outcome
    Given a source session with one synthetic turn
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report marks the turn as imported
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report marks every turn as skipped unchanged
    Given the source turn's reply changed
    And the gateway fails the next trace verification
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 1
    And the report marks the turn as failed
    Given a tracking target that fails the capability check
    When I rerun the import through the CLI with fake dependencies
    Then the process exits with status 2
    And the report carries the safe error "mlflow_capability_unsupported"
