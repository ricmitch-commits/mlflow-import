"""Runtime MLflow capability probe for the importer's writer boundary.

The writer is authorized by Task 0's pinned compatibility profile
(``tests/integration/mlflow_profile.py``, recorded in
``docs/compatibility.md``): the pinned client/server pair behaviorally
proved experiment lookup/creation, context-free span creation and
completion, full-payload retrieval, paginated search, public trace tag
update, retirement, and named trace artifacts. This module adapts that
recorded profile into the runtime gate the importer runs before any
mutation.

:func:`probe_capabilities` is strictly READ-ONLY in both modes. It inspects
client methods, module-level public API availability and signatures, the
read-only tracking-server `/version` attestation, server reachability and
protocol responses, the recorded compatibility profile (exact installed
client/server versions against the pinned profile), the configured artifact
capability, and the reported payload limits. It never
creates a probe experiment, trace, or artifact, never uploads, and never
deletes: the mutation-based proofs belong only to Task 0's disposable
compatibility-test environment, so the facts that cannot be re-proven
without mutation (``LiveSpan.end``, the named-artifact composition's
server-side behavior, retirement semantics, and the 1 MiB inline round-trip
floor) are taken from the recorded profile and apply only while both the
installed client and attested server match the pinned profile version. A
numeric version match alone can never make an unsupported pair pass: the
read-only checks must also succeed, and the probe stays behavioral wherever
behavior can be observed without mutation.

Mode semantics: a NORMAL write (``dry_run=False``) always sets
``needs_retirement=True`` in the effective request regardless of the
initial action plan, because rollback is required after any partial create.
Capability failure aborts the entire invocation before mutation through
:func:`require_capabilities`, which raises a safe
:class:`~opencode_mlflow.errors.CapabilityError` (``exit_status == 2``).
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import get_artifact_repository

from opencode_mlflow.errors import CapabilityError

__all__ = [
    "PINNED_MAX_INLINE_PAYLOAD_BYTES",
    "PINNED_MLFLOW_VERSION",
    "VERSION_ATTESTATION_TIMEOUT_SECONDS",
    "MAX_VERSION_RESPONSE_BYTES",
    "PROBE_EXPERIMENT_NAME",
    "PROBE_TRACE_ID",
    "REQUIRED_METHODS",
    "VersionTransport",
    "CapabilityReport",
    "CapabilityRequest",
    "attest_server_version",
    "version_endpoint",
    "probe_capabilities",
    "require_capabilities",
]

#: The pinned client version from Task 0's recorded compatibility profile.
#: The runtime probe trusts the profile's mutation-based evidence only
#: while both the installed client and attested tracking server match this
#: version exactly; a newer pair requires a new profile before the package
#: constraint changes.
PINNED_MLFLOW_VERSION = "3.17.2"

#: Bound the read-only server attestation. The endpoint is checked before any
#: profile-backed behavior is trusted.
VERSION_ATTESTATION_TIMEOUT_SECONDS = 5.0
MAX_VERSION_RESPONSE_BYTES = 64 * 1024

#: The inline payload floor the pinned profile verified to round-trip
#: byte-exact through span inputs/outputs (larger payloads must become named
#: trace artifacts). Reported as the target's payload limit when the
#: recorded profile applies.
PINNED_MAX_INLINE_PAYLOAD_BYTES = 1_048_576

#: The pinned public operations the writer depends on — identical to the
#: profile's ``REQUIRED_METHODS`` (pinned by the gateway contract test).
REQUIRED_METHODS: frozenset[str] = frozenset(
    {
        "server_version_attestation",
        "get_experiment_by_name",
        "create_experiment",
        "mlflow.start_span_no_context",
        "LiveSpan.end",
        "get_trace",
        "search_traces",
        "delete_traces",
        "upload_named_json",
        "verify_named_json",
    }
)

#: Client methods the probe checks for availability (the pinned surface
#: also includes the public trace tag update methods, which the profile
#: proved; the gateway records verified artifact digests on traces
#: through ``set_trace_tag``).
_CLIENT_OPERATIONS: tuple[str, ...] = (
    "get_experiment_by_name",
    "create_experiment",
    "get_trace",
    "search_traces",
    "set_trace_tag",
    "delete_trace_tag",
    "delete_traces",
)

#: Keywords ``mlflow.start_span_no_context`` must accept for the writer.
_SPAN_API_KEYWORDS: tuple[str, ...] = (
    "name",
    "span_type",
    "parent_span",
    "inputs",
    "attributes",
    "tags",
    "metadata",
    "experiment_id",
    "start_time_ns",
)

#: Operations whose proof is mutation-based and therefore belongs to the
#: recorded profile; they apply only while the installed client version
#: matches the pin (and, for the artifact composition, while the public
#: ``mlflow.artifacts.get_artifact_repository`` factory is available).
_PROFILE_BACKED_OPERATIONS: frozenset[str] = frozenset({"LiveSpan.end"})


VersionTransport = Callable[[str], object]
"""Test-only read-only transport receiving the client's base tracking URI."""


class _VersionAttestationFailure(Exception):
    """Internal marker for a safe, non-leaking attestation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


#: Reserved names for the read-only reachability/protocol probes. The
#: experiment name must not exist on a healthy target; the trace id must
#: raise a proper MLflow protocol error (an error carrying a machine-
#: readable ``error_code``, like a real ``MlflowException``) rather than a
#: transport failure.
PROBE_EXPERIMENT_NAME = "opencode-mlflow-capability-probe"
PROBE_TRACE_ID = "tr-opencode-mlflow-capability-probe"

_INCOMPATIBLE_MESSAGE = "the MLflow tracking target failed the required capability check"


@dataclass(frozen=True)
class CapabilityRequest:
    """What one invocation needs from the tracking target.

    ``needs_retirement`` is the initial action plan's requirement; a normal
    write (``dry_run=False``) always forces it to ``True`` because rollback
    is required after any partial create. ``needs_named_artifacts`` marks an
    invocation that will upload named trace artifacts.
    """

    needs_retirement: bool = False
    needs_named_artifacts: bool = False


@dataclass(frozen=True)
class CapabilityReport:
    """What :func:`probe_capabilities` proved about the target, read-only.

    Carries the profile report's fields — ``required_methods`` (only
    operations whose read-only verification or pinned-profile evidence
    succeeded), the reported inline payload limit, named-artifact and
    retirement support, and a ``safe_error`` safe to display to a CLI user
    — plus the effective request (after the normal-mode retirement
    normalization) and the probed mode.
    """

    is_compatible: bool
    required_methods: frozenset[str]
    max_inline_payload_bytes: int
    named_artifact_supported: bool
    retirement_supported: bool
    needs_retirement: bool
    needs_named_artifacts: bool
    dry_run: bool
    safe_error: str | None


def probe_capabilities(
    client: MlflowClient,
    request: CapabilityRequest,
    dry_run: bool,
    *,
    version_transport: VersionTransport | None = None,
) -> CapabilityReport:
    """Probe the target read-only and report whether it satisfies the profile.

    Never mutates the target in either mode: no probe experiment, trace, or
    artifact is created, nothing is uploaded or deleted. Facts that need a
    mutation to prove are taken from the recorded compatibility profile and
    apply only while the installed client and attested server match
    :data:`PINNED_MLFLOW_VERSION`; if they cannot be proven that way the
    report is incompatible and :func:`require_capabilities` fails the whole
    invocation with ``mlflow_capability_unsupported`` (exit status ``2``).
    """
    errors: list[str] = []
    verified: set[str] = set()
    needs_retirement = bool(request.needs_retirement) or not dry_run
    needs_named_artifacts = bool(request.needs_named_artifacts)

    # 1. The recorded compatibility profile applies only to the pinned
    #    client version; a version mismatch is a safe, loud failure.
    installed = getattr(mlflow, "__version__", None)
    profile_applies = installed == PINNED_MLFLOW_VERSION
    if not profile_applies:
        errors.append(
            f"installed mlflow client version {installed!r} does not match the pinned compatibility profile "
            f"version {PINNED_MLFLOW_VERSION!r}"
        )

    # The client version is not evidence about the tracking server. The
    # production transport performs the documented read-only GET; fake
    # clients must inject this seam explicitly.
    server_version_attested = False
    try:
        server_version = attest_server_version(
            getattr(client, "tracking_uri", None),
            transport=version_transport,
        )
        if server_version != PINNED_MLFLOW_VERSION:
            errors.append("the tracking server version does not match the pinned compatibility profile")
        else:
            server_version_attested = True
            verified.add("server_version_attestation")
    except _VersionAttestationFailure as exc:
        errors.append(exc.reason)
    except Exception:  # noqa: BLE001 - never expose transport/library details
        errors.append("tracking server version attestation failed")

    profile_backed = profile_applies and server_version_attested

    # 2. Module-level public trace API: availability plus signature shape.
    start_span: Any = getattr(mlflow, "start_span_no_context", None)
    if not callable(start_span):
        errors.append("the public module-level span API (mlflow.start_span_no_context) is not available")
    else:
        missing = _missing_span_keywords(start_span)
        if missing:
            errors.append(
                "mlflow.start_span_no_context does not accept required parameters: " + ", ".join(sorted(missing))
            )
        else:
            verified.add("mlflow.start_span_no_context")

    # 3. Client method availability.
    for name in _CLIENT_OPERATIONS:
        if callable(getattr(client, name, None)):
            verified.add(name)
        else:
            errors.append(f"the tracking client does not provide the required method {name!r}")

    # 4. The public artifact-repository factory behind the named-artifact
    #    composition (Ruling 5: never the semi-private mlflow.store import).
    composition_available = callable(get_artifact_repository)

    # 5. Server reachability and protocol responses, read-only.
    try:
        found = client.get_experiment_by_name(PROBE_EXPERIMENT_NAME)
        if found is not None:
            errors.append("the reserved capability-probe experiment name already exists on the target")
    except Exception as exc:  # noqa: BLE001 - a failed probe is a report input, never a crash
        errors.append(f"server reachability probe failed: {type(exc).__name__}")
    try:
        client.get_trace(PROBE_TRACE_ID, display=False)
        errors.append("retrieving an unknown trace id did not raise a protocol error")
    except Exception as exc:  # noqa: BLE001 - a failed probe is a report input, never a crash
        if getattr(exc, "error_code", None) is None:
            errors.append(f"trace retrieval did not produce an MLflow protocol error: {type(exc).__name__}")

    # 6. Profile-backed operations: the mutation-proven facts (span
    #    completion, the named-artifact composition's server behavior, the
    #    inline payload floor) apply only to the pinned, version-matched
    #    pair with the public composition available.
    if profile_backed:
        verified.update(_PROFILE_BACKED_OPERATIONS)
        if composition_available:
            verified.update({"upload_named_json", "verify_named_json"})

    # Method presence alone does not establish the pinned server retirement
    # semantics, so retirement is gated by the exact server attestation too.
    retirement_supported = profile_backed and "delete_traces" in verified
    named_artifact_supported = {"upload_named_json", "verify_named_json"} <= verified
    max_inline_payload_bytes = PINNED_MAX_INLINE_PAYLOAD_BYTES if profile_backed else 0

    if needs_retirement and not retirement_supported:
        errors.append("trace retirement is required for this invocation but is not available")
    if needs_named_artifacts and not named_artifact_supported:
        errors.append("named trace artifacts are required for this invocation but are not available")

    is_compatible = not errors and REQUIRED_METHODS <= verified
    return CapabilityReport(
        is_compatible=is_compatible,
        required_methods=frozenset(verified),
        max_inline_payload_bytes=max_inline_payload_bytes,
        named_artifact_supported=named_artifact_supported,
        retirement_supported=retirement_supported,
        needs_retirement=needs_retirement,
        needs_named_artifacts=needs_named_artifacts,
        dry_run=dry_run,
        safe_error="; ".join(errors) if errors else None,
    )


def version_endpoint(tracking_uri: str) -> str:
    """Return the documented read-only tracking-server version endpoint.

    Only HTTP(S) tracking servers can satisfy the profile. URI userinfo,
    query parameters, fragments, and credentials are never included in the
    resulting request.
    """
    if not isinstance(tracking_uri, str) or not tracking_uri.strip():
        raise _VersionAttestationFailure("tracking server version attestation requires an HTTP(S) tracking URI")
    parsed = urllib.parse.urlsplit(tracking_uri)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise _VersionAttestationFailure("tracking server version attestation requires an HTTP(S) tracking URI")
    try:
        host = parsed.hostname
        assert host is not None
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parsed.port
    except ValueError:
        raise _VersionAttestationFailure("tracking server version attestation requires a valid HTTP(S) URI") from None
    netloc = host if port is None else f"{host}:{port}"
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/version" if base_path else "/version"
    # Query strings and fragments are client-side configuration, not part of
    # the tracking server's static URL prefix. In particular, a query may
    # carry a credential; never forward it to the /version request.
    return urllib.parse.urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def attest_server_version(
    tracking_uri: str | None,
    *,
    transport: VersionTransport | None = None,
) -> str:
    """Perform and validate the exact read-only tracking-server attestation.

    ``transport`` is an explicit dependency-injection seam. The default uses
    the documented ``GET <tracking-base>/version`` endpoint and standard
    MLflow/requests-compatible environment settings.
    """
    if transport is not None:
        try:
            response = transport(str(tracking_uri) if tracking_uri is not None else "")
        except _VersionAttestationFailure:
            raise
        except Exception:
            raise _VersionAttestationFailure("tracking server version attestation failed") from None
    else:
        response = _http_version_transport(tracking_uri)
    payload = _decode_version_response(response)
    version = payload.get("version")
    if not isinstance(version, str) or not version:
        raise _VersionAttestationFailure("tracking server version attestation returned a malformed response")
    return version


def _decode_version_response(response: object) -> Mapping[str, object]:
    """Decode an injected or HTTP response without retaining unsafe details."""
    _enforce_version_response_size(response)
    payload: object = response
    if isinstance(response, (bytes, bytearray, str)):
        try:
            text = bytes(response).decode("utf-8").strip() if not isinstance(response, str) else response.strip()
        except UnicodeDecodeError:
            raise _VersionAttestationFailure(
                "tracking server version attestation returned a malformed response"
            ) from None
        # MLflow's documented endpoint currently returns the version as a
        # plain-text semantic version (not JSON) on the pinned server line.
        if re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", text):
            return {"version": text}
        try:
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _VersionAttestationFailure(
                "tracking server version attestation returned a malformed response"
            ) from None
    if not isinstance(payload, Mapping):
        raise _VersionAttestationFailure("tracking server version attestation returned a malformed response")
    return payload


def _enforce_version_response_size(response: object) -> None:
    """Reject oversized injected responses before parsing or trusting fields."""
    size: int | None = None
    if isinstance(response, (bytes, bytearray)):
        size = len(response)
    elif isinstance(response, str):
        size = len(response.encode("utf-8"))
    elif isinstance(response, Mapping):
        try:
            size = len(json.dumps(response, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError, OverflowError):
            raise _VersionAttestationFailure(
                "tracking server version attestation returned a malformed response"
            ) from None
    if size is not None and size > MAX_VERSION_RESPONSE_BYTES:
        raise _VersionAttestationFailure("tracking server version attestation returned a malformed response")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail closed instead of forwarding authorization to a redirected host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _http_version_transport(tracking_uri: str | None) -> Mapping[str, object]:
    """Read the server version through stdlib HTTP and standard env config."""
    endpoint = version_endpoint(tracking_uri or "")
    headers = {"Accept": "application/json", "User-Agent": "opencode-mlflow-capability-probe"}
    token = os.environ.get("MLFLOW_TRACKING_TOKEN")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username is not None or password is not None:
        raw = f"{username or ''}:{password or ''}".encode()
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")

    request = urllib.request.Request(endpoint, headers=headers, method="GET")
    context = ssl.create_default_context()
    insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").strip().lower() in {"1", "true", "yes", "on"}
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    else:
        ca_path = (
            os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
        )
        if ca_path:
            if os.path.isdir(ca_path):
                context.load_verify_locations(capath=ca_path)
            else:
                context.load_verify_locations(cafile=ca_path)
    client_cert_path = os.environ.get("MLFLOW_TRACKING_CLIENT_CERT_PATH")
    if client_cert_path:
        context.load_cert_chain(certfile=client_cert_path)

    opener = urllib.request.build_opener(_NoRedirectHandler(), urllib.request.HTTPSHandler(context=context))
    try:
        with opener.open(request, timeout=VERSION_ATTESTATION_TIMEOUT_SECONDS) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            status = int(status_value)
            if status < 200 or status >= 300:
                raise _VersionAttestationFailure("tracking server version endpoint returned an unexpected response")
            body = response.read(MAX_VERSION_RESPONSE_BYTES + 1)
    except _VersionAttestationFailure:
        raise
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, ssl.SSLError):
        raise _VersionAttestationFailure("tracking server version attestation failed") from None
    if len(body) > MAX_VERSION_RESPONSE_BYTES:
        raise _VersionAttestationFailure("tracking server version attestation returned a malformed response")
    return _decode_version_response(body)


def require_capabilities(report: CapabilityReport) -> None:
    """Abort the invocation before any mutation when the report is incompatible.

    Raises a safe :class:`~opencode_mlflow.errors.CapabilityError` with code
    ``mlflow_capability_unsupported`` and exit status ``2``; the message
    carries only the probe's safe error causes (no user values, paths, or
    transcript content).
    """
    if not report.is_compatible:
        raise CapabilityError(
            "mlflow_capability_unsupported",
            report.safe_error or _INCOMPATIBLE_MESSAGE,
        )


def _missing_span_keywords(start_span: Any) -> tuple[str, ...]:
    """Required keywords missing from the span API's resolved signature."""
    try:
        parameters = inspect.signature(start_span).parameters
    except (TypeError, ValueError):  # pragma: no cover - unresolvable signature
        return _SPAN_API_KEYWORDS
    return tuple(keyword for keyword in _SPAN_API_KEYWORDS if keyword not in parameters)
