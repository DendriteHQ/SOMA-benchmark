"""Scoped exec broker for the Copilot SWE sandbox.

The agent must never hold credentials for the isolated Docker daemon. That daemon has to run
``--privileged`` for a nested dockerd to work at all, which makes access to its API a
privilege boundary rather than a sandbox: it must be treated as fully trusted.

This broker is the only holder of the dind client certificates. It exposes exactly one
capability - run a command inside the single, pre-created SWE sandbox container - so a
misbehaving or prompt-injected agent cannot create containers, pull images, or touch
anything else in the isolated daemon.

Protocol (deliberately minimal so the client is a few lines of shell):
    POST /exec
        Authorization: Bearer <token>
        Content-Type: text/plain
        body: the shell command to run inside the sandbox
    -> 200, body = stdout followed by stderr, headers:
        X-Exit-Code: <int>          command's exit status (124 timed out, 125 unknown)
        X-Timed-Out: true|false
        X-Output-Truncated: true|false
    GET /healthz -> 200 "ok"

Exit codes are never guessed. A zero reported here means the daemon told us the process
exited zero; every "we could not determine it" path reports UNKNOWN_EXIT_CODE instead.
Getting that wrong is worse than it sounds in a benchmark harness: an agent that is told its
test run passed stops working and submits whatever it has.

The container id is fixed by configuration and is NOT accepted from the request: the
client chooses *what to run*, never *where*.
"""

from __future__ import annotations

import hmac
import json
import os
import shlex
import ssl
import sys
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A single test run can legitimately produce a lot of output (full pytest logs), but an
# unbounded read lets one command exhaust broker memory. When the cap is hit the middle is
# dropped rather than the tail: a test runner puts its summary - the part the agent actually
# needs - at the end.
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
OUTPUT_HEAD_BYTES = MAX_OUTPUT_BYTES // 2
OUTPUT_TAIL_BYTES = MAX_OUTPUT_BYTES - OUTPUT_HEAD_BYTES
READ_CHUNK_BYTES = 256 * 1024
MAX_COMMAND_BYTES = 128 * 1024
DEFAULT_EXEC_TIMEOUT_SECONDS = 900.0
TIMEOUT_EXIT_CODE = 124
# Mirrors the docker CLI's own use of 125 for "the command could not be run/observed", and
# is deliberately not 0: see the module docstring.
UNKNOWN_EXIT_CODE = 125
# The output stream is drained to EOF before this is polled, so the process has all but
# certainly exited already; this only covers the brief window before the daemon publishes
# the exec's final state.
EXIT_STATE_POLL_ATTEMPTS = 40
EXIT_STATE_POLL_SLEEP_SECONDS = 0.25


def _require_env(name: str) -> str:
    value = str(os.environ.get(name, "")).strip()
    if not value:
        raise SystemExit(f"swe-exec-broker: required environment variable {name} is missing")
    return value


class BrokerConfig:
    def __init__(self) -> None:
        self.token = _require_env("SWE_EXEC_BROKER_TOKEN")
        self.container = _require_env("SWE_EXEC_SANDBOX_CONTAINER")
        self.dind_host = _require_env("SWE_EXEC_DIND_HOST")
        self.dind_port = int(os.environ.get("SWE_EXEC_DIND_PORT", "2376"))
        self.cert_dir = str(os.environ.get("SWE_EXEC_DIND_CERT_DIR", "/certs/client"))
        self.repo_path = str(os.environ.get("SWE_EXEC_SANDBOX_REPO_PATH", "/testbed"))
        # Discovered once at sandbox startup (see _discover_swe_sandbox_env_bin_dir in
        # copilot.py) and prepended to PATH for every exec'd command below. SWE-bench task
        # images almost universally put the task's real interpreter/deps in a conda env
        # named "testbed" rather than on the default PATH, and that env is not activated in
        # a plain non-interactive shell (activation is a bash function sourced from
        # conda.sh, which a one-off `sh -lc`/`bash -lc` never runs). Without this, every
        # agent independently rediscovers the same path by trial and error each run. Empty
        # when discovery found nothing usable - PATH is then left untouched.
        self.env_bin_dir = str(os.environ.get("SWE_EXEC_SANDBOX_ENV_BIN_DIR", "")).strip()
        self.port = int(os.environ.get("SWE_EXEC_BROKER_PORT", "8100"))
        self.timeout = float(
            os.environ.get("SWE_EXEC_TIMEOUT_SECONDS", DEFAULT_EXEC_TIMEOUT_SECONDS)
        )


def _build_ssl_context(cert_dir: str) -> ssl.SSLContext:
    ca_path = os.path.join(cert_dir, "ca.pem")
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")
    for path in (ca_path, cert_path, key_path):
        if not os.path.isfile(path):
            raise SystemExit(f"swe-exec-broker: dind client certificate missing at {path}")

    # Hostname verification stays on: dind's generated server certificate carries a SAN for
    # "docker", and compose publishes that alias for the daemon, so SWE_EXEC_DIND_HOST must
    # be "docker" rather than the compose service name. The CA pin authenticates the daemon
    # (only dind's own CA can sign it); the client certificate authenticates this broker.
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_path)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return context


class _CappedBuffer:
    """Accumulates a stream, keeping its head and tail and dropping the middle."""

    def __init__(self, head_bytes: int, tail_bytes: int) -> None:
        self._head = bytearray()
        self._tail = bytearray()
        self._head_cap = head_bytes
        self._tail_cap = tail_bytes
        self.dropped = 0

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        if len(self._head) < self._head_cap:
            take = self._head_cap - len(self._head)
            self._head += chunk[:take]
            chunk = chunk[take:]
            if not chunk:
                return
        self._tail += chunk
        excess = len(self._tail) - self._tail_cap
        if excess > 0:
            del self._tail[:excess]
            self.dropped += excess

    def value(self) -> bytes:
        if not self.dropped:
            return bytes(self._head + self._tail)
        marker = (
            f"\n\n[swe-exec: {self.dropped} bytes of output omitted here; "
            "the head and tail of the stream are shown]\n\n"
        ).encode("utf-8")
        return bytes(self._head) + marker + bytes(self._tail)


def _read_exec_output(response, *, deadline: float) -> tuple[bytes, bytes, int]:
    """Read Docker's hijacked exec stream to EOF, returning (stdout, stderr, dropped).

    Read to EOF, always: the stream closing is the only signal that the process finished, so
    abandoning it early - which is what a bare ``read(cap)`` does on a chatty test run - both
    truncates the output and leaves the exec still running, with no exit code to inspect
    afterwards. Output past the cap is therefore consumed and discarded rather than left
    unread; only the *retained* bytes are bounded.

    Frame layout: 1 byte stream id, 3 padding bytes, 4 bytes big-endian length, payload.
    Framing is decided once, from the first header: the exec is created with ``Tty: false``
    so a multiplexed stream is what we expect, and a stream that does not start with a valid
    header is passed through verbatim rather than discarded.
    """
    stdout = _CappedBuffer(OUTPUT_HEAD_BYTES, OUTPUT_TAIL_BYTES)
    stderr = _CappedBuffer(OUTPUT_HEAD_BYTES, OUTPUT_TAIL_BYTES)
    buffer = bytearray()
    framed: bool | None = None

    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("exec output exceeded the allotted time")
        chunk = response.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        if framed is False:
            stdout.append(chunk)
            continue

        buffer += chunk
        if framed is None:
            if len(buffer) < 8:
                continue
            framed = buffer[0] in (0, 1, 2) and buffer[1:4] == b"\x00\x00\x00"
            if not framed:
                stdout.append(bytes(buffer))
                buffer.clear()
                continue

        while len(buffer) >= 8:
            size = int.from_bytes(buffer[4:8], "big")
            if len(buffer) < 8 + size:
                break
            stream_id = buffer[0]
            payload = bytes(buffer[8:8 + size])
            del buffer[:8 + size]
            if stream_id == 2:
                stderr.append(payload)
            else:
                stdout.append(payload)

    if buffer:
        # A trailing partial frame - keep whatever payload arrived instead of dropping it,
        # and never let it invalidate the frames already decoded.
        stdout.append(bytes(buffer[8:]) if framed else bytes(buffer))

    return stdout.value(), stderr.value(), stdout.dropped + stderr.dropped


class DindClient:
    def __init__(self, config: BrokerConfig) -> None:
        self._config = config
        self._ssl_context = _build_ssl_context(config.cert_dir)

    def _connect(self, *, timeout: float) -> http.client.HTTPSConnection:
        return http.client.HTTPSConnection(
            self._config.dind_host,
            self._config.dind_port,
            timeout=timeout,
            context=self._ssl_context,
        )

    def _request(
        self,
        connection: http.client.HTTPSConnection,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> tuple[int, bytes]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read(MAX_OUTPUT_BYTES)

    def _wait_for_exit_code(self, exec_id: str, *, timeout: float) -> int:
        """Ask the daemon for the exec's real exit status, or report it as unknown.

        ``ExitCode`` is null for as long as ``Running`` is true, and the value must never be
        defaulted to 0 in that case - nor when the inspect call itself fails. A caller that
        is told "0" acts on it.
        """
        for _ in range(EXIT_STATE_POLL_ATTEMPTS):
            connection = self._connect(timeout=timeout)
            try:
                status, raw = self._request(connection, "GET", f"/exec/{exec_id}/json")
            finally:
                connection.close()
            if status != 200:
                raise RuntimeError(
                    f"dind refused exec inspect (HTTP {status}): "
                    f"{raw.decode('utf-8', errors='replace').strip()}"
                )
            parsed = json.loads(raw.decode("utf-8"))
            exit_code = parsed.get("ExitCode")
            if not parsed.get("Running") and isinstance(exit_code, int):
                return exit_code
            time.sleep(EXIT_STATE_POLL_SLEEP_SECONDS)
        return UNKNOWN_EXIT_CODE

    def exec_command(self, command: str, *, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        effective_command = command
        if self._config.env_bin_dir:
            # Quoted so a bin dir containing spaces still works; PATH is prepended (not
            # replaced) so the rest of the image's PATH - and any absolute path the agent
            # used anyway - keeps working unchanged.
            effective_command = (
                f'export PATH={shlex.quote(self._config.env_bin_dir)}":$PATH"; ' + command
            )
        connection = self._connect(timeout=timeout)
        try:
            status, raw = self._request(
                connection,
                "POST",
                f"/containers/{self._config.container}/exec",
                {
                    "AttachStdout": True,
                    "AttachStderr": True,
                    "Tty": False,
                    "WorkingDir": self._config.repo_path,
                    # bash, not sh: SWE-bench task images are set up with the harness's own
                    # bash eval scripts in mind, and agents routinely reach for bash-only
                    # syntax (`source activate ...`) that plain sh/dash rejects outright.
                    "Cmd": ["bash", "-lc", effective_command],
                },
            )
            if status != 201:
                raise RuntimeError(
                    f"dind refused exec creation (HTTP {status}): "
                    f"{raw.decode('utf-8', errors='replace').strip()}"
                )
            exec_id = str(json.loads(raw.decode("utf-8")).get("Id", "")).strip()
            if not exec_id:
                raise RuntimeError("dind did not return an exec id")
        finally:
            connection.close()

        # A fresh connection: the start call hijacks the stream, so it must not be reused.
        # The response is streamed rather than read in one shot - see _read_exec_output.
        connection = self._connect(timeout=timeout)
        try:
            body = json.dumps({"Detach": False, "Tty": False}).encode("utf-8")
            connection.request(
                "POST",
                f"/exec/{exec_id}/start",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(
                    f"dind refused exec start (HTTP {response.status}): "
                    f"{response.read(MAX_OUTPUT_BYTES).decode('utf-8', errors='replace').strip()}"
                )
            stdout, stderr, dropped = _read_exec_output(response, deadline=deadline)
        finally:
            connection.close()

        exit_code = self._wait_for_exit_code(exec_id, timeout=timeout)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "dropped": dropped,
        }


class BrokerHandler(BaseHTTPRequestHandler):
    server_version = "soma-swe-exec-broker/1.0"
    config: BrokerConfig
    client: DindClient

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        sys.stderr.write("[swe-exec-broker] %s\n" % (format % args))

    def _reply(self, status: int, body: bytes, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = str(self.headers.get("Authorization", ""))
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        # Constant-time compare: the miner's compression service shares the agent network,
        # so the token is what stops it from driving the sandbox and faking a solution.
        return hmac.compare_digest(header[len(prefix):], self.config.token)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        if self.path == "/healthz":
            self._reply(200, b"ok")
            return
        self._reply(404, b"not found\n")

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if self.path != "/exec":
            self._reply(404, b"not found\n")
            return
        if not self._authorized():
            self._reply(401, b"unauthorized\n")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(400, b"invalid Content-Length\n")
            return
        if length <= 0:
            self._reply(400, b"empty command\n")
            return
        if length > MAX_COMMAND_BYTES:
            self._reply(413, b"command too large\n")
            return

        command = self.rfile.read(length).decode("utf-8", errors="replace")
        timeout = self.config.timeout
        raw_timeout = self.headers.get("X-Timeout-Seconds")
        if raw_timeout:
            try:
                timeout = min(float(raw_timeout), self.config.timeout)
            except ValueError:
                pass

        try:
            result = self.client.exec_command(command, timeout=timeout)
        except TimeoutError:
            # Covers both a stalled socket and the wall-clock cap in _read_exec_output.
            # Note the command itself is NOT killed - the exec keeps running in the sandbox.
            self._reply(
                200,
                b"swe-exec: command timed out (it may still be running in the sandbox)\n",
                {
                    "X-Exit-Code": str(TIMEOUT_EXIT_CODE),
                    "X-Timed-Out": "true",
                    "X-Output-Truncated": "true",
                },
            )
            return
        except OSError as exc:
            # A transport failure against the daemon, not a timeout: TimeoutError above
            # already covers socket timeouts (socket.timeout is an alias for it).
            self._reply(
                200,
                f"swe-exec: sandbox exec failed: {exc}\n".encode("utf-8"),
                {
                    "X-Exit-Code": str(UNKNOWN_EXIT_CODE),
                    "X-Timed-Out": "false",
                    "X-Output-Truncated": "false",
                },
            )
            return
        except Exception as exc:  # noqa: BLE001 - report any broker/daemon fault to the caller
            self._reply(
                500,
                f"swe-exec: broker error: {exc}\n".encode("utf-8"),
                {"X-Exit-Code": str(UNKNOWN_EXIT_CODE), "X-Timed-Out": "false"},
            )
            return

        body = result["stdout"] + result["stderr"]
        if result["exit_code"] == UNKNOWN_EXIT_CODE:
            # Say so in the body too: the agent reads output far more reliably than it reads
            # exit codes, and this is exactly the case where it must not assume success.
            body += (
                b"\nswe-exec: the command's exit status could not be determined "
                b"(it may still be running in the sandbox) - do not read this as success\n"
            )
        self._reply(
            200,
            body,
            {
                "X-Exit-Code": str(result["exit_code"]),
                "X-Timed-Out": "false",
                "X-Output-Truncated": "true" if result["dropped"] else "false",
            },
        )


def main() -> None:
    config = BrokerConfig()
    BrokerHandler.config = config
    BrokerHandler.client = DindClient(config)
    server = ThreadingHTTPServer(("0.0.0.0", config.port), BrokerHandler)
    sys.stderr.write(
        f"[swe-exec-broker] listening on :{config.port}, "
        f"sandbox={config.container} repo={config.repo_path}\n"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
