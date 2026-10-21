"""Documentation verification for the opencode-mlflow docs suite.

Checks the documentation against the shipped code, in both directions:

1. **Paths** — every markdown link and every backticked repository path in
   ``README.md``, ``CHANGELOG.md``, and ``docs/*.md`` exists on disk.
2. **CLI** — every option documented in ``README.md``/``docs/usage.md``
   exists in the real parser (and vice versa), and every example command
   in those files parses successfully through ``parse_config``.
3. **Result schema** — the documented top-level sections, summary fields,
   session fields, turn fields, action enum, reasons, and ownership keys
   match the serializer in ``reporting.py``/``reconciliation.py`` exactly.
4. **Schema catalog** — the table/column matrix documented in
   ``docs/schema-catalog-v1.md`` mirrors ``source/catalog.py`` exactly
   (per generation, per table, both directions).
5. **Security** — the documented detector categories, replacement
   markers, protected identity fields, and excluded tables match the code.
6. **Constants** — documented error codes and pinned limits are compared
   against the constants and string literals in ``src/``.

Run from the repository root through the project venv:

    .venv/bin/python scripts/verify_docs.py

Exit status 0 means every check passed; each failure is printed with the
document and the offending item.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from opencode_mlflow.config import (  # noqa: E402
    DEFAULT_DB_PATH,
    LOG_LEVEL_CHOICES,
    OUTPUT_CHOICES,
    build_parser,
    parse_config,
)
from opencode_mlflow.content.artifacts import DEFAULT_INLINE_LIMIT_BYTES  # noqa: E402
from opencode_mlflow.content.redaction import (  # noqa: E402
    DETECTOR_CATEGORIES,
    REDACTED_BINARY_MARKER,
    REDACTED_MARKER,
)
from opencode_mlflow.contracts import PROTECTED_IDENTITY_FIELDS, RedactionMode  # noqa: E402
from opencode_mlflow.mlflow.capabilities import (  # noqa: E402
    PINNED_MAX_INLINE_PAYLOAD_BYTES,
    PINNED_MLFLOW_VERSION,
)
from opencode_mlflow.mlflow.gateway import (  # noqa: E402
    DEFAULT_UPLOAD_CHUNK_BYTES,
    OWNERSHIP_IMPORTER_VALUE,
    OWNERSHIP_TAG_KEY,
    TRACE_METADATA_VALUE_LIMIT,
    TRACE_TAG_VALUE_LIMIT,
    VERIFY_CHUNK_BYTES,
)
from opencode_mlflow.mlflow.reconciliation import (  # noqa: E402
    ACTION_IMPORTED,
    ACTION_SKIPPED,
    ARTIFACT_PATH_TAG,
    ARTIFACT_SHA256_TAG,
    ARTIFACT_SIZE_TAG,
    OWNERSHIP_CATALOG_VERSION_KEY,
    OWNERSHIP_IMPORTER_VERSION_KEY,
    OWNERSHIP_MAPPING_VERSION_KEY,
    OWNERSHIP_OPERATION_KEY,
    OWNERSHIP_REDACTION_MODE_KEY,
    OWNERSHIP_RENDER_FINGERPRINT_KEY,
    OWNERSHIP_SESSION_KEY,
    OWNERSHIP_SOURCE_FINGERPRINT_KEY,
    OWNERSHIP_SOURCE_SCHEMA_KEY,
    OWNERSHIP_TURN_INDEX_KEY,
    OWNERSHIP_TURN_KEY,
    REASON_SESSION_TOMBSTONE,
    REASON_STALE_TURN,
    REASON_UNCHANGED,
)
from opencode_mlflow.reporting import (  # noqa: E402
    ACTION_SET,
    REASON_DRY_RUN,
    REASON_NOT_SELECTED,
    RESULT_SCHEMA,
    ReportBuilder,
    TurnResult,
)
from opencode_mlflow.source.catalog import (  # noqa: E402
    OPENCODE_SCHEMA_CATALOG_V1,
    SCHEMA_CATALOG_VERSION,
    SECRET_TABLES,
    SESSION_SHARE_TABLE,
)

FAILURES: list[str] = []

DOC_FILES = [
    REPO / "README.md",
    REPO / "CHANGELOG.md",
    *sorted((REPO / "docs").glob("*.md")),
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(document: str, message: str) -> None:
    FAILURES.append(f"{document}: {message}")


def human_bytes(size: int) -> str:
    """The documented spelling of one byte-size constant (``16 MiB``...)."""
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MiB"
    if size % 1024 == 0:
        return f"{size // 1024} KiB"
    return str(size)


# ---------------------------------------------------------------------------
# 1. Referenced paths
# ---------------------------------------------------------------------------

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_REPOSITORY_PATH = re.compile(r"`((?:\.\./)?(?:src|tests|docs|\.github|scripts)/[^`\s]+)`")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)


def check_paths() -> None:
    for doc in DOC_FILES:
        text = read(doc)
        for target in _MARKDOWN_LINK.findall(text):
            if _SCHEME.match(target) or target.startswith("#"):
                continue
            resolved = (doc.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                fail(doc.name, f"markdown link target does not exist: {target}")
        for span in _REPOSITORY_PATH.findall(text):
            # Backticked repository paths are written repo-root-relative
            # (``src/...``, ``tests/...``) even inside ``docs/``; markdown
            # links are document-relative. Accept either resolution.
            if not any((base / span).resolve().exists() for base in (doc.parent, REPO)):
                fail(doc.name, f"referenced repository path does not exist: {span}")


# ---------------------------------------------------------------------------
# 2. CLI options and example commands
# ---------------------------------------------------------------------------


def documented_options(text: str) -> set[str]:
    """Every `--option` mentioned anywhere in the document.

    A backticked span may carry a placeholder after the flag name
    (`` `--experiment-name NAME` ``); only the flag itself is collected.
    """
    return set(re.findall(r"`(--[a-z-]+)[^`]*`", text))


def example_commands(text: str) -> list[list[str]]:
    """Every `opencode-mlflow ...` command inside a code fence, as words.

    Continuation lines (trailing backslash) join the command; a `|`
    pipe ends it. Commands containing a bracketed placeholder such as
    ``[options]`` are syntax sketches and are skipped.
    """
    commands: list[list[str]] = []
    fence = False
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if buffer:
                commands.append(_shell_words(" ".join(buffer)))
                buffer = []
            fence = not fence
            continue
        if not fence or stripped.startswith("#"):
            continue
        if stripped.startswith("opencode-mlflow"):
            buffer = [stripped]
        elif buffer:
            buffer.append(stripped)
        else:
            continue
        if not stripped.endswith("\\"):
            commands.append(_shell_words(" ".join(buffer)))
            buffer = []
    if buffer:
        commands.append(_shell_words(" ".join(buffer)))
    return [command for command in commands if command and "[" not in " ".join(command)]


def _shell_words(joined: str) -> list[str]:
    words: list[str] = []
    for piece in joined.split():
        if piece == "|":
            break
        if piece == "\\":
            continue
        words.append(piece.removesuffix("\\"))
    return words


def _iter_parsers(parser: argparse.ArgumentParser) -> Iterator[argparse.ArgumentParser]:
    """The parser and every subparser it declares (options live on both)."""
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_parsers(subparser)


def check_cli() -> None:
    parser = build_parser()
    parser_options = {
        option
        for subparser in _iter_parsers(parser)
        for action in subparser._actions
        for option in action.option_strings
        if option.startswith("--")
    }

    for name in ("README.md", "docs/usage.md"):
        text = read(REPO / name)
        documented = documented_options(text)
        if missing := documented - parser_options:
            fail(name, f"documented options absent from the parser: {sorted(missing)}")
        if extra := parser_options - documented:
            fail(name, f"parser options missing from the documentation: {sorted(extra)}")
        # Choices appear as ``human\|json`` inside option tables, so a
        # plain substring check is used; the example commands below prove
        # the values themselves parse.
        for choice in OUTPUT_CHOICES:
            if choice not in text:
                fail(name, f"output choice {choice!r} not documented")
        for choice in LOG_LEVEL_CHOICES:
            if choice not in text:
                fail(name, f"log-level choice {choice!r} not documented")
        if DEFAULT_DB_PATH not in text:
            fail(name, f"default database path {DEFAULT_DB_PATH!r} not documented")

    for name in ("README.md", "docs/usage.md"):
        for command in example_commands(read(REPO / name)):
            if command[0] != "opencode-mlflow":
                continue
            try:
                parse_config(command[1:])
            except Exception as error:  # noqa: BLE001 - report every failure shape
                fail(name, f"example command rejected by parse_config: {' '.join(command)} ({error})")


# ---------------------------------------------------------------------------
# 3. Result schema
# ---------------------------------------------------------------------------


def table_first_column(text: str, heading: str) -> list[str]:
    """First-column backticked names of the first table after a heading."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(heading)), None)
    if start is None:
        return []
    rows: list[str] = []
    in_table = False
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            if not set(line) <= {"|", "-", " ", ":"}:
                cell = line.strip("|").split("|")[0].strip()
                match = re.fullmatch(r"`([a-z_][a-z0-9_.]*)`", cell)
                if match:
                    rows.append(match.group(1))
            in_table = True
        elif in_table:
            break
    return rows


def check_result_schema() -> None:
    text = read(REPO / "docs/result-schema.md")

    builder = ReportBuilder(
        dry_run=True, experiment_name="verify-docs", redaction_mode=RedactionMode.VERBATIM, generated_ns=1
    )
    builder.set_experiment("0", "resolved")
    builder.set_source(schema_generation="current", fingerprint="0" * 64)
    builder.set_source_counts(sessions_scanned=1, sessions_selected=1)
    builder.set_retired_traces(0)
    builder.set_unexamined_owned_traces(0)
    builder.add_session_scan("ses_verify", selected=True)
    builder.add_turn(
        TurnResult(
            session_id="ses_verify",
            turn_key="turn-" + "0" * 64,
            action=ACTION_SKIPPED,
            reason=REASON_DRY_RUN,
            planned_action=ACTION_IMPORTED,
        )
    )
    document = builder.build()

    sections = {
        "## Top-level sections": set(document),
        "## Summary": set(document["summary"]),
        "## Session entries": set(document["sessions"][0]),
        "## Turn entries": set(document["sessions"][0]["turns"][0]),
    }
    for heading, actual in sections.items():
        documented = set(table_first_column(text, heading))
        if documented != actual:
            fail("result-schema.md", f"{heading}: documented {sorted(documented)} != serializer {sorted(actual)}")

    enum_text = text.split("## Action enum")[1].split("```text")[1].split("```")[0]
    documented_enum = {word.strip() for word in enum_text.replace("\n", " ").split("|")}
    if documented_enum != set(ACTION_SET):
        fail("result-schema.md", f"action enum {sorted(documented_enum)} != code {sorted(ACTION_SET)}")

    actual_reasons = {
        REASON_UNCHANGED,
        REASON_DRY_RUN,
        REASON_NOT_SELECTED,
        REASON_STALE_TURN,
        REASON_SESSION_TOMBSTONE,
    }
    documented_reasons = set(table_first_column(text, "## Reasons"))
    if documented_reasons != actual_reasons:
        fail("result-schema.md", f"reasons {sorted(documented_reasons)} != code {sorted(actual_reasons)}")

    actual_keys = {
        OWNERSHIP_TAG_KEY,
        OWNERSHIP_SESSION_KEY,
        OWNERSHIP_TURN_INDEX_KEY,
        OWNERSHIP_TURN_KEY,
        OWNERSHIP_SOURCE_FINGERPRINT_KEY,
        OWNERSHIP_SOURCE_SCHEMA_KEY,
        OWNERSHIP_IMPORTER_VERSION_KEY,
        OWNERSHIP_MAPPING_VERSION_KEY,
        OWNERSHIP_REDACTION_MODE_KEY,
        OWNERSHIP_RENDER_FINGERPRINT_KEY,
        OWNERSHIP_OPERATION_KEY,
        OWNERSHIP_CATALOG_VERSION_KEY,
        ARTIFACT_PATH_TAG,
        ARTIFACT_SHA256_TAG,
        ARTIFACT_SIZE_TAG,
    }
    documented_keys = set(table_first_column(text, "## Ownership metadata and artifact identity"))
    if documented_keys != actual_keys:
        fail("result-schema.md", f"ownership keys {sorted(documented_keys)} != code {sorted(actual_keys)}")

    if RESULT_SCHEMA not in text:
        fail("result-schema.md", f"result schema id {RESULT_SCHEMA!r} missing")


# ---------------------------------------------------------------------------
# 4. Schema catalog
# ---------------------------------------------------------------------------

_GENERATION_SECTIONS = {
    "legacy": "## Legacy generation (`legacy`)",
    "current": "## Current generation (`current`)",
}


def _generation_text(text: str, heading: str) -> str:
    """One `##` generation section (excluding nested `###` boundaries' content)."""
    start = text.index(heading)
    match = re.search(r"^## ", text[start + len(heading) :], re.MULTILINE)
    return text[start : start + len(heading) + match.start()] if match else text[start:]


def _table_section(generation_text: str, table: str) -> str:
    """The `### `table`` subsection inside one generation."""
    match = re.search(rf"###\s+`{table}`[^\n]*\n(.*?)(?=\n###|\n##|\Z)", generation_text, re.DOTALL)
    return match.group(1) if match else ""


def _identifiers(section: str) -> set[str]:
    """Every backticked snake_case identifier in a section."""
    return set(re.findall(r"`([a-z_][a-z0-9_]*)`", section))


def _paren_columns(scoped_text: str, table: str) -> set[str]:
    """Column names the allowlist lists inside `table(...)`."""
    columns: set[str] = set()
    for match in re.finditer(rf"`{table}\(([^`]*)\)`", scoped_text, re.DOTALL):
        columns.update(re.findall(r"[a-z_][a-z0-9_]*", match.group(1)))
    return columns


def check_schema_catalog() -> None:
    text = read(REPO / "docs/schema-catalog-v1.md")
    if SCHEMA_CATALOG_VERSION not in text:
        fail("schema-catalog-v1.md", f"catalog version {SCHEMA_CATALOG_VERSION!r} missing")
    for table in sorted(SECRET_TABLES):
        if table not in text:
            fail("schema-catalog-v1.md", f"excluded table {table!r} missing")
    if SESSION_SHARE_TABLE not in text:
        fail("schema-catalog-v1.md", "session_share omission missing")

    allowlist_start = text.index("## Session-scoped allowlists")
    allowlist_match = re.search(r"^## ", text[allowlist_start + 2 :], re.MULTILINE)
    allowlist_end = allowlist_start + 2 + allowlist_match.start() if allowlist_match else len(text)
    allowlists = text[allowlist_start:allowlist_end]
    legacy_allowlist, _, current_allowlist = allowlists.partition("\nCurrent:")

    for catalog in OPENCODE_SCHEMA_CATALOG_V1:
        generation = _generation_text(text, _GENERATION_SECTIONS[catalog.generation])
        for table, columns in catalog.tables.items():
            documented = _identifiers(_table_section(generation, table))
            actual = {column.name for column in columns}
            if documented != actual:
                fail(
                    "schema-catalog-v1.md",
                    f"[{catalog.generation}] {table}: documented {sorted(documented)} != catalog {sorted(actual)}",
                )
        scoped = legacy_allowlist if catalog.generation == "legacy" else current_allowlist
        for table, columns in catalog.session_scoped_tables.items():
            documented = _paren_columns(scoped, table)
            actual = {column.name for column in columns}
            if documented != actual:
                fail(
                    "schema-catalog-v1.md",
                    f"[{catalog.generation}] {table}: documented {sorted(documented)} != catalog {sorted(actual)}",
                )
        for table, columns in {**catalog.tables, **catalog.session_scoped_tables}.items():
            for column in columns:
                if not column.required:
                    marked = any(f"`{column.name}`" in line and "optional" in line for line in generation.splitlines())
                    if not marked:
                        fail(
                            "schema-catalog-v1.md",
                            f"[{catalog.generation}] {table}.{column.name} is optional but never marked optional",
                        )


# ---------------------------------------------------------------------------
# 5. Security constants
# ---------------------------------------------------------------------------


def check_security() -> None:
    text = read(REPO / "docs/security.md")
    documented = set(table_first_column(text, "## Detector categories and pattern semantics"))
    if documented != set(DETECTOR_CATEGORIES):
        fail("security.md", f"detector categories {sorted(documented)} != code {sorted(DETECTOR_CATEGORIES)}")
    for marker in (REDACTED_MARKER, REDACTED_BINARY_MARKER):
        if marker not in text:
            fail("security.md", f"replacement marker {marker!r} missing")
    protected_section = text.split("## Protected identifiers")[1].split("\n## ")[0]
    documented_protected = set(re.findall(r"`([a-z_][a-z0-9_]*)`", protected_section))
    if documented_protected != set(PROTECTED_IDENTITY_FIELDS):
        fail(
            "security.md",
            f"protected identity fields {sorted(documented_protected)} != code {sorted(PROTECTED_IDENTITY_FIELDS)}",
        )
    for table in sorted(SECRET_TABLES):
        if table not in text:
            fail("security.md", f"excluded table {table!r} missing")


# ---------------------------------------------------------------------------
# 6. Documented codes and constants
# ---------------------------------------------------------------------------


def check_codes_and_constants() -> None:
    suite_text = "\n".join(read(doc) for doc in DOC_FILES)
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO / "src").rglob("*.py")))

    for name in ("docs/result-schema.md", "docs/operations.md"):
        text = read(REPO / name)
        candidates: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and not set(stripped) <= {"|", "-", " ", ":"}:
                first_cell = stripped.strip("|").split("|")[0]
                candidates.update(re.findall(r"`([a-z][a-z0-9_]{2,})`", first_cell))
        for code in sorted(candidates):
            if f'"{code}"' not in source_text:
                fail(Path(name).name, f"documented code {code!r} appears nowhere in src/")

    expectations = [
        (PINNED_MLFLOW_VERSION, "pinned MLflow version"),
        (PINNED_MAX_INLINE_PAYLOAD_BYTES, "pinned inline payload floor"),
        (DEFAULT_INLINE_LIMIT_BYTES, "conservative inline payload limit"),
        (DEFAULT_UPLOAD_CHUNK_BYTES, "upload chunk size"),
        (VERIFY_CHUNK_BYTES, "verification chunk size"),
        (TRACE_TAG_VALUE_LIMIT, "trace tag truncation limit"),
        (TRACE_METADATA_VALUE_LIMIT, "trace metadata truncation limit"),
        (OWNERSHIP_TAG_KEY, "ownership marker key"),
        (OWNERSHIP_IMPORTER_VALUE, "ownership marker value"),
    ]
    for constant, description in expectations:
        if str(constant) not in suite_text and human_bytes(constant) not in suite_text:
            fail("docs suite", f"{description} ({constant}) not documented with its exact value")


def main() -> int:
    checks = [
        ("referenced paths", check_paths),
        ("CLI options and examples", check_cli),
        ("result schema", check_result_schema),
        ("schema catalog", check_schema_catalog),
        ("security constants", check_security),
        ("codes and constants", check_codes_and_constants),
    ]
    for name, check in checks:
        try:
            check()
        except Exception:  # noqa: BLE001 - a broken check is a failure, not a crash
            FAILURES.append(f"{name}: check raised an exception\n{traceback.format_exc()}")
    if FAILURES:
        print(f"docs verification FAILED ({len(FAILURES)} problem(s)):")
        for problem in FAILURES:
            print(f"  - {problem}")
        return 1
    print("docs verification passed: paths, CLI, result schema, schema catalog, security, constants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
