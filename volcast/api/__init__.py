"""volcast.api - FastAPI backend exposing the forecasting library over HTTP."""

from __future__ import annotations

__all__ = ["create_app", "app"]


def __getattr__(name: str):
    # Keep ``import volcast.api`` cheap: FastAPI is an optional extra.
    if name in __all__:
        from .main import app, create_app

        return {"create_app": create_app, "app": app}[name]
    raise AttributeError(name)
