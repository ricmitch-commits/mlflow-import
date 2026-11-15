"""LOCAL-only proxy qualification; never honors an ambient tracking server."""

import hashlib
import json
import sqlite3
import ssl
import subprocess
import threading
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.artifacts import get_artifact_repository

from mlflow_import import resources
from mlflow_import.config import ImportConfig
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.capabilities import _VersionAttestationFailure, attest_server_version
from mlflow_import.mlflow.gateway import MlflowGateway
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.integration.test_resource_quota import (
    test_phase_two_quota_exhaustion_preserves_prior_and_recovers_without_leaks as quota_case,
)
from tests.integration.test_session_recovery import (
    test_later_turn_failure_preserves_entire_prior_session_and_filtered_pending as recovery_case,
)
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tls_servers(tmp_path_factory):
    """Two real isolated CAs and their loopback-only HTTPS servers, entirely offline."""
    root = tmp_path_factory.mktemp("artifact-tls")
    extensions = root / "server.ext"
    extensions.write_text(
        "subjectAltName=IP:127.0.0.1\nextendedKeyUsage=serverAuth\n"
        "keyUsage=critical,digitalSignature\nbasicConstraints=critical,CA:FALSE\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n"
    )
    servers = []
    payload = b"synthetic TLS artifact"

    def openssl(*args):
        subprocess.run(["openssl", *map(str, args)], check=True, capture_output=True, timeout=15)

    def start_server(ca_name):
        ca, ca_key = root / f"{ca_name}.pem", root / f"{ca_name}.key"
        cert, key, csr = (root / f"{ca_name}-server.{suffix}" for suffix in ("pem", "key", "csr"))
        openssl(
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-subj",
            f"/CN={ca_name}",
            "-days",
            "2",
            "-keyout",
            ca_key,
            "-out",
            ca,
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-addext",
            "subjectKeyIdentifier=hash",
        )
        openssl(
            "req",
            "-new",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-subj",
            "/CN=127.0.0.1",
            "-keyout",
            key,
            "-out",
            csr,
        )
        openssl(
            "x509",
            "-req",
            "-in",
            csr,
            "-CA",
            ca,
            "-CAkey",
            ca_key,
            "-set_serial",
            "2",
            "-days",
            "2",
            "-extfile",
            extensions,
            "-out",
            cert,
        )
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                authorization = self.headers.get("Authorization")
                received.append((self.path, authorization))
                if authorization != "Bearer synthetic-tls-token":
                    self.send_error(401)
                    return
                if self.path == "/version":
                    body = b"3.17.2"
                elif "?" in self.path:
                    body = json.dumps(
                        {"files": [{"path": "body.json", "is_dir": False, "file_size": len(payload)}]}
                    ).encode()
                else:
                    body = payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        result = SimpleNamespace(
            ca=ca,
            uri=f"https://127.0.0.1:{server.server_port}",
            received=received,
            server=server,
            thread=thread,
        )
        servers.append(result)
        return result

    try:
        selected, default = start_server("selected-ca"), start_server("default-ca")
        ca_directory = root / "selected-ca-directory"
        ca_directory.mkdir()
        (ca_directory / "ca.pem").write_bytes(selected.ca.read_bytes())
        openssl("rehash", ca_directory)
        empty_directory = root / "empty-default-ca-directory"
        empty_directory.mkdir()
        yield selected, default, ca_directory, empty_directory, payload
    finally:
        for server in servers:
            server.server.shutdown()
            server.server.server_close()
            server.thread.join(timeout=5)
            assert not server.thread.is_alive()


@pytest.mark.parametrize(
    "variable, directory",
    [
        ("MLFLOW_TRACKING_SERVER_CERT_PATH", False),
        ("REQUESTS_CA_BUNDLE", False),
        ("CURL_CA_BUNDLE", False),
        ("MLFLOW_TRACKING_SERVER_CERT_PATH", True),
    ],
)
def test_explicit_ca_replaces_default_trust_for_authenticated_artifact_and_version_gets(
    tls_servers, monkeypatch, tmp_path, variable, directory
):
    selected, default, ca_directory, empty_directory, payload = tls_servers
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "MLFLOW_TRACKING_AUTH",
        "MLFLOW_TRACKING_AWS_SIGV4",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_TRACKING_INSECURE_TLS",
        "MLFLOW_TRACKING_SERVER_CERT_PATH",
        "MLFLOW_TRACKING_CLIENT_CERT_PATH",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("SSL_CERT_FILE", str(default.ca))
    monkeypatch.setenv("SSL_CERT_DIR", str(empty_directory))
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "synthetic-tls-token")

    # A real handshake proves this server IS trusted by the default context.
    # The old additive implementation consequently accepted both CAs.
    opener, headers = resources.artifact_http_configuration()
    with opener.open(urllib.request.Request(default.uri + "/version", headers=headers), timeout=5) as response:
        assert response.read(64) == b"3.17.2"
    default.received.clear()
    monkeypatch.setenv(variable, str(ca_directory if directory else selected.ca))

    assert attest_server_version(selected.uri) == "3.17.2"
    with pytest.raises(_VersionAttestationFailure, match="attestation failed"):
        attest_server_version(default.uri)

    selected_repository = SimpleNamespace(artifact_uri=selected.uri + resources.ARTIFACT_PROXY_PATH + "/experiment")
    default_repository = SimpleNamespace(artifact_uri=default.uri + resources.ARTIFACT_PROXY_PATH + "/experiment")
    assert resources.artifact_exists(selected_repository, "body.json")
    with resources.temporary_storage(1024 * 1024):
        with resources.temporary_directory(prefix="authenticated-tls-") as root:
            downloaded = resources.copy_artifact(selected_repository, "body.json", root, len(payload))
            assert downloaded.read_bytes() == payload
            with pytest.raises(ItemImportError, match="HTTP artifact read failed"):
                resources.copy_artifact(default_repository, "body.json", root, len(payload))
    with pytest.raises(ItemImportError, match="HTTP artifact read failed"):
        resources.artifact_exists(default_repository, "body.json")
    assert default.received == []  # Neither HTTP requests nor credentials crossed the failed handshake.
    assert all(auth == "Bearer synthetic-tls-token" for _, auth in selected.received)


@pytest.fixture(scope="module")
def proxy_server():
    with live_mlflow_server(requested_uri="", proxy_artifacts=True) as server:
        yield server


def test_remote_fresh_unchanged_append_dry_run(proxy_server, tmp_path, monkeypatch):
    source = build_current_source_db(
        tmp_path,
        sessions=[("proxy", 1000)],
        messages=[
            ("u1", "proxy", "user", 1, '{"text":"synthetic prompt"}'),
            ("a1", "proxy", "assistant", 2, '{"text":"synthetic reply"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("proxy"), source, tracking_uri=proxy_server.uri)
    dependencies = build_dependencies(config)
    mutations = []
    require = MlflowGateway._require_mutation_allowed

    def count_mutations(self):
        mutations.append(True)
        return require(self)

    monkeypatch.setattr(MlflowGateway, "_require_mutation_allowed", count_mutations)
    dry = run_import(replace(config, dry_run=True), dependencies)
    assert exit_status(dry) == 0, dry.report
    assert proxy_server.client.get_experiment_by_name(config.experiment_name) is None
    assert mutations == []
    dry.close()
    first = run_import(config, dependencies)
    assert exit_status(first) == 0, first.report
    experiment_id = first.report["experiment"]["id"]
    prior = first.report["sessions"][0]["turns"][0]["trace_id"]
    experiment = proxy_server.client.get_experiment(experiment_id)
    assert experiment.artifact_location.startswith("mlflow-artifacts:")
    first.close()
    mutations.clear()
    unchanged = run_import(config, dependencies)
    assert exit_status(unchanged) == 0, unchanged.report
    assert unchanged.report["sessions"][0]["turns"][0]["trace_id"] == prior
    assert unchanged.report["sessions"][0]["turns"][0]["reason"] == "unchanged"
    assert mutations == []
    unchanged.close()
    with sqlite3.connect(source) as db:
        db.executemany(
            "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("u2", "proxy", "user", 3, 3, 3, '{"text":"synthetic append"}'),
                ("a2", "proxy", "assistant", 4, 4, 4, '{"text":"synthetic append reply"}'),
            ],
        )
    appended = run_import(config, dependencies)
    assert exit_status(appended) == 0, appended.report
    assert appended.report["sessions"][0]["turns"][0]["trace_id"] == prior
    assert len(appended.report["sessions"][0]["turns"]) == 2
    appended.close()
    mutations.clear()
    before = {t.trace_id for t in proxy_server.client.search_traces(locations=[experiment_id], include_spans=False)}
    dry = run_import(replace(config, dry_run=True), dependencies)
    assert exit_status(dry) == 0, dry.report
    dry.close()
    after = {t.trace_id for t in proxy_server.client.search_traces(locations=[experiment_id], include_spans=False)}
    assert before == after and len(after) == 2
    assert mutations == []


def test_proxy_recovery_and_filter_sensitivity(proxy_server, tmp_path):
    recovery_case(proxy_server, tmp_path)


@pytest.mark.parametrize("phase", ["serialization", "download"])
def test_proxy_shared_quota_preserves_prior_and_recovers(proxy_server, tmp_path, monkeypatch, phase):
    quota_case(proxy_server, tmp_path, monkeypatch, phase)


@pytest.mark.parametrize("scheme", ["mlflow", "http"])
def test_public_repository_listing_download_checksum_and_quota(proxy_server, tmp_path, scheme):
    name = unique_experiment_name("proxy-wire")
    experiment_id = proxy_server.client.create_experiment(name)
    experiment = proxy_server.client.get_experiment(experiment_id)
    repository = get_artifact_repository(experiment.artifact_location, tracking_uri=proxy_server.uri)
    if scheme == "http":
        repository = get_artifact_repository(repository.artifact_uri, tracking_uri=proxy_server.uri)
    payload = b"synthetic proxy bytes\n" * 10000
    source = tmp_path / "body.json"
    source.write_bytes(payload)
    repository.log_artifact(str(source), artifact_path="test")
    with resources.temporary_storage(1024 * 1024) as budget:
        assert resources.artifact_exists(repository, "test/body.json")
        assert not resources.artifact_exists(repository, "test/missing.json")
        entries = list(resources.iter_artifacts(repository, "test"))
        assert [(entry.path, entry.file_size) for entry in entries] == [("test/body.json", len(payload))]
        with resources.temporary_directory(prefix="proxy-") as root:
            path = resources.copy_artifact(repository, "test/body.json", root, len(payload))
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(payload).digest()
            with pytest.raises(ItemImportError):
                resources.copy_artifact(repository, "test/body.json", root, len(payload) - 1)
        gateway = MlflowGateway(proxy_server.client)
        gateway.verify_experiment_file(
            experiment_id, "test/body.json", hashlib.sha256(payload).hexdigest(), len(payload)
        )
        with pytest.raises(ItemImportError, match="checksum"):
            gateway.verify_experiment_file(experiment_id, "test/body.json", "0" * 64, len(payload))
        assert budget.peak_bytes <= budget.max_bytes
    with resources.temporary_storage(64 * 1024):
        with resources.temporary_directory(prefix="proxy-") as root:
            with pytest.raises(resources.TemporaryStorageError):
                resources.copy_artifact(repository, "test/body.json", root, len(payload))
