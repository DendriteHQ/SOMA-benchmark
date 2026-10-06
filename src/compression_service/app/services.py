"""Compressor services, service side of `soma-compressor-services/1`.

A miner runs in a worker thread. When it calls `services.jev.decide(...)` the call is
parked: the HTTP handler that is waiting on the session returns the call to the proxy
as `pending`, the proxy executes it, and `POST /transform/resume` hands the result back
to the parked thread. The miner sees an ordinary blocking call.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any, Callable

PROTOCOL = "soma-compressor-services/1"

#: How long a parked call waits for its result past the offer's deadline before it
#: gives up with `timeout`. The proxy enforces the deadline; this only keeps a miner
#: from hanging when the proxy has stopped resuming (it timed out, or crashed).
WAIT_MARGIN_S = 5.0

#: Sessions nobody resumed are dropped after this long.
SESSION_TTL_S = 300.0


class ServiceError(Exception):
    """A service call failed. The miner should carry on without the answer."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


class ServiceUnavailable(ServiceError):
    """The service is not offered on this turn."""


class _Jev:
    def __init__(self, services: "Services") -> None:
        self._services = services

    def decide(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        result = self.decide_many([{"state": state, "questions": questions, "model": model}])[0]
        if isinstance(result, ServiceError):
            raise result
        return result

    def decide_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any] | ServiceError]:
        offer = self._services.offer("jev")
        if offer is None:
            raise ServiceUnavailable("not_offered", "jev is not offered on this turn")
        calls = []
        for item in requests:
            body = {
                "model": item.get("model") or offer.get("default_model"),
                "state": item.get("state"),
                "questions": item.get("questions"),
            }
            calls.append({"service": "jev", "op": "decide", "request": body})
        return self._services._round(calls)


class Services:
    """What a miner receives as `services`."""

    ServiceError = ServiceError
    ServiceUnavailable = ServiceUnavailable

    def __init__(self, offer: dict[str, Any], round_trip: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None) -> None:
        self._offer = offer if isinstance(offer, dict) else {}
        self._round_trip = round_trip
        self._lock = threading.Lock()
        self._next_id = 0
        self.jev = _Jev(self)

    def offers(self, name: str) -> bool:
        return self._round_trip is not None and isinstance(self._offer.get(name), dict)

    def offer(self, name: str) -> dict[str, Any] | None:
        return dict(self._offer[name]) if self.offers(name) else None

    def _round(self, calls: list[dict[str, Any]]) -> list[dict[str, Any] | ServiceError]:
        if self._round_trip is None:
            raise ServiceUnavailable("not_offered", "no services on this turn")
        if not calls:
            return []
        with self._lock:  # one round at a time, whichever thread asks
            ids = []
            for call in calls:
                self._next_id += 1
                call["id"] = str(self._next_id)
                ids.append(call["id"])
            by_id = {r.get("id"): r for r in self._round_trip(calls)}
        out: list[dict[str, Any] | ServiceError] = []
        for cid in ids:
            r = by_id.get(cid)
            if not isinstance(r, dict):
                out.append(ServiceError("invalid_result", f"no result for call {cid}"))
            elif r.get("ok") and isinstance(r.get("response"), dict):
                out.append(r["response"])
            else:
                err = r.get("error") if isinstance(r.get("error"), dict) else {}
                out.append(ServiceError(str(err.get("code") or "upstream_error"), str(err.get("message") or "")))
        return out


class Session:
    """One suspended miner invocation and the queues that connect it to HTTP handlers."""

    def __init__(self, offer: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex
        self.created = time.monotonic()
        self.request: Any = None  # the /transform request that started it
        self.round = 0
        self._events: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._pending_ids: set[str] = set()
        deadline_ms = max((o.get("deadline_ms") or 0) for o in offer.values() if isinstance(o, dict)) if offer else 0
        self._wait_s = deadline_ms / 1000.0 + WAIT_MARGIN_S
        self.services = Services(offer, self._round_trip)

    # -- miner thread -------------------------------------------------------------

    def _round_trip(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.round += 1
        self._pending_ids = {c["id"] for c in calls}
        self._events.put(("pending", {"session": self.id, "round": self.round, "calls": calls}))
        try:
            return self._results.get(timeout=max(self._wait_s - (time.monotonic() - self.created), 0.1))
        except queue.Empty:
            self._pending_ids = set()
            return [{"id": c["id"], "ok": False, "error": {"code": "timeout", "message": "no result from the proxy"}}
                    for c in calls]

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        self._events.put(("error", error) if error is not None else ("done", result))

    # -- HTTP handlers --------------------------------------------------------------

    def next_event(self) -> tuple[str, Any]:
        return self._events.get()

    def deliver(self, results: Any) -> None:
        if not isinstance(results, list) or {r.get("id") for r in results if isinstance(r, dict)} != self._pending_ids:
            raise ValueError("results must answer exactly the calls of the last round")
        self._pending_ids = set()
        self._results.put(results)


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def add(self, session: Session) -> None:
        with self._lock:
            now = time.monotonic()
            for sid in [s for s, v in self._sessions.items() if now - v.created > SESSION_TTL_S]:
                del self._sessions[sid]
            self._sessions[session.id] = session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(sid)

    def drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)


def parse_offer(services: Any) -> dict[str, Any] | None:
    """The offer of a `/transform` request, or None when it offers nothing we speak."""
    if not isinstance(services, dict) or services.get("protocol") != PROTOCOL:
        return None
    offer = services.get("offer")
    return offer if isinstance(offer, dict) and offer else None
