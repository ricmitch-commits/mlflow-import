"""Safe error hierarchy for the OpenCode → MLflow importer.

Every failure the importer reports carries a stable machine-readable
``code``, a ``safe_message`` sanitized for logs (no user values, source
paths, credentials, or transcript content), and the ``exit_status`` the
process should return. ``str()`` of any error yields only the safe message,
so error text can be logged directly without a separate sanitization pass;
codes and exit statuses are read from the structured attributes instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opencode_mlflow.mlflow.gateway import CreatedTrace


class ImporterError(Exception):
    """Base class for importer failures.

    ``exit_status`` defaults to the subclass's ``default_exit_status``
    (configuration/source/capability/snapshot failures exit ``2``; per-item
    and reconciliation failures exit ``1``) and may be overridden per error.
    """

    default_exit_status = 2

    def __init__(self, code: str, safe_message: str, exit_status: int | None = None) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.exit_status = self.default_exit_status if exit_status is None else exit_status
        self.cleanup_warnings: tuple[str, ...] = ()

    def __str__(self) -> str:
        # Only the safe message: the string form must never carry the code
        # or any unsafe context.
        return self.safe_message

    def attach_cleanup_warnings(self, warnings: tuple[str, ...]) -> None:
        """Attach safe cleanup state observed while unwinding any failure."""
        self.cleanup_warnings = warnings


class ConfigurationError(ImporterError):
    """Invalid invocation or configuration; fails before any source or MLflow work."""

    default_exit_status = 2


class SourceCompatibilityError(ImporterError):
    """The source database schema or content is unsupported."""

    default_exit_status = 2


class SnapshotError(ImporterError):
    """A consistent read-only source snapshot could not be established."""

    default_exit_status = 2

    def __init__(
        self,
        code: str,
        safe_message: str,
        exit_status: int | None = None,
        *,
        cleanup_warnings: tuple[str, ...] = (),
    ) -> None:
        super().__init__(code, safe_message, exit_status)
        self.cleanup_warnings = cleanup_warnings

    def attach_cleanup_warnings(self, warnings: tuple[str, ...]) -> None:
        """Attach only the safe cleanup state observed while unwinding."""
        self.cleanup_warnings = warnings


class CapabilityError(ImporterError):
    """The tracking server lacks a capability required for the requested mode."""

    default_exit_status = 2


class ItemImportError(ImporterError):
    """A single session/turn failed while other items may still proceed."""

    default_exit_status = 1


class ReconciliationConflict(ImporterError):
    """A previously imported trace could not be reconciled safely."""

    default_exit_status = 1


class PartialTraceCreateError(ItemImportError):
    """A failed create with known identity, requiring bounded rollback."""

    def __init__(self, created: CreatedTrace, cause: Exception) -> None:
        code = cause.code if isinstance(cause, ImporterError) else "trace_create_failed"
        super().__init__(code, "trace creation failed after the root was created")
        self.created = created
        self.rollback_retired = False
