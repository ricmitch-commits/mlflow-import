Feature: Turn-to-trace mapping and content fidelity
  Each selected logical OpenCode turn maps to exactly one importer-owned
  MLflow trace, preserving ordering, roles, and payload bytes in verbatim
  mode. The mapping-seam scenarios pin the pure mapping contract; the
  boundary scenarios drive the same six-turn fixture through the full CLI
  pipeline and assert what the tracking target actually stored.

  Scenario: A six-turn session maps to six owned trace specs
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    Then the mapping produces 6 trace specs
    And every spec carries the native session id "ses-synthetic-1"
    And the specs carry 6 distinct client request ids
    And every spec metadata records "opencode.session_id" as "ses-synthetic-1"
    And every spec metadata records "opencode.importer" as "opencode-mlflow"

  Scenario: Root values follow the first prompt and the final result
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    Then the root input of turn 1 is the first user prompt
    And the root output of turn 1 is the final assistant text part
    And the root output of turn 2 combines 2 ordered text parts
    And the root output of turn 3 is omitted

  Scenario: Source outcomes map to authoritative root statuses
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    Then turn 1 maps to root status "OK" with outcome "assistant_result"
    And turn 3 maps to root status "UNSET" with outcome "lifecycle_observed"
    And turn 4 maps to root status "ERROR" with outcome "explicit_failure"

  Scenario: Typed children preserve the turn's source records
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    Then turn 1 has 7 children
    And turn 1 has a child of span type "LLM" with source id "msg-synthetic-a1"
    And turn 1 has a child of span type "TOOL" with source id "msg-synthetic-a1#content-1"
    And turn 1 has a child of span type "UNKNOWN" with source id "msg-synthetic-a1#content-2"
    And turn 1 has an event child with source id "msg-synthetic-a1#content-0"
    And the child of turn 1 with source id "msg-synthetic-a1#content-1" has parent source id "msg-synthetic-a1"
    And the child of turn 1 with source id "msg-synthetic-u1#content-0" has the trace root as parent

  Scenario: Children stream lazily with fresh cursors
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    Then iterating the children of turn 1 twice yields the same 7 children

  Scenario: Mapping is deterministic per turn
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec twice
    Then both mappings produce identical specs

  Scenario: Binding generated identities preserves the artifact fingerprints
    Given the synthetic six-turn session fixture
    When I map every replayed turn to a trace spec
    And I bind generated identities for turn 1
    Then the artifact snapshot fingerprint of turn 1 is unchanged
    And the bound identity map records the trace and span ids

  Scenario: A six-turn session imports as six traces under one native session id
    Given the synthetic six-turn session source database
    When I run an import through the CLI with fake dependencies
    Then the process exits with status 0
    And the report records 6 imported turns for session "ses-synthetic-1"
    And the gateway stored 6 traces for the native session id "ses-synthetic-1"
    And the imported traces carry 6 distinct trace ids
    And the stored root statuses follow the source outcomes

  Scenario: Root values and children round-trip through the gateway
    Given the synthetic six-turn session source database
    When I run an import through the CLI with fake dependencies
    Then the root input of imported turn 1 is the first user prompt
    And the root output of imported turn 1 is the final assistant text part
    And the root output of imported turn 3 is omitted
    And imported turn 1 has a child of span type "LLM" with source id "msg-synthetic-a1"
    And imported turn 1 has a child of span type "TOOL" with source id "msg-synthetic-a1#content-1"
    And imported turn 1 has a child of span type "UNKNOWN" with source id "msg-synthetic-u1#content-0"
    And the child of imported turn 1 with source id "msg-synthetic-u1#content-0" is attached to the trace root
