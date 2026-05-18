"""
bloomberg/_session.py
Gestion de la session Bloomberg -- singleton thread-safe.

Une seule connexion est maintenue par processus Python. Les fonctions
bdh/bdp/bds l'utilisent toutes sans recréer de session à chaque appel.
"""
from __future__ import annotations

import threading
from typing import Optional

import blpapi  # type: ignore[import-untyped]

REFDATA_SVC = "//blp/refdata"


class BloombergSession:
    """Encapsule une session blpapi ouverte sur //blp/refdata."""

    def __init__(self, host: str = "localhost", port: int = 8194) -> None:
        opts = blpapi.SessionOptions()
        opts.setServerHost(host)
        opts.setServerPort(port)

        self._session = blpapi.Session(opts)

        if not self._session.start():
            raise ConnectionError(
                "Impossible de démarrer la session Bloomberg. "
                "Bloomberg Terminal doit être ouvert et connecté."
            )
        if not self._session.openService(REFDATA_SVC):
            raise ConnectionError(
                f"Impossible d'ouvrir le service {REFDATA_SVC}"
            )

        self._svc = self._session.getService(REFDATA_SVC)

    def create_request(self, request_type: str) -> blpapi.Request:
        """Crée une requête Bloomberg du type indiqué."""
        return self._svc.createRequest(request_type)

    def send(self, request: blpapi.Request) -> None:
        """Envoie une requête au serveur Bloomberg."""
        self._session.sendRequest(request)

    def next_event(self) -> blpapi.Event:
        """Attend et retourne le prochain événement Bloomberg."""
        return self._session.nextEvent()

    def stop(self) -> None:
        """Ferme proprement la session."""
        self._session.stop()


# ───────────────────────────── Singleton global ──────────────────────────────

_lock = threading.Lock()
_instance: Optional[BloombergSession] = None


def get_session(host: str = "localhost", port: int = 8194) -> BloombergSession:
    """
    Retourne la session Bloomberg globale.
    La crée lors du premier appel (lazy init), la réutilise ensuite.

    Parameters
    ----------
    host : hôte Bloomberg (défaut: 'localhost' pour DAPI local)
    port : port Bloomberg (défaut: 8194)
    """
    global _instance
    with _lock:
        if _instance is None:
            _instance = BloombergSession(host, port)
    return _instance


def close() -> None:
    """Ferme et libère la session Bloomberg globale."""
    global _instance
    with _lock:
        if _instance is not None:
            _instance.stop()
            _instance = None
