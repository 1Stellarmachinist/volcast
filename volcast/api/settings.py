"""API settings via pydantic-settings.

BUG #8 FIX: ``cookie_secure`` is derived from ``is_prod`` instead of being
hardcoded True. A Secure cookie is silently dropped by the browser over plain
http, so on localhost the refresh cookie simply never came back and every session
looked expired after 15 minutes.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOLCAST_",
        env_file=".env",
        extra="ignore",
    )

    env: str = "dev"                     # dev | prod
    app_name: str = "volcast"

    # --- WebAuthn / passkeys -------------------------------------------------
    rp_id: str = "localhost"
    rp_name: str = "volcast"
    # Origins that may complete a WebAuthn ceremony (comma separated).
    expected_origins: str = "http://localhost:5173,http://localhost:8000"

    # --- Tokens --------------------------------------------------------------
    jwt_secret: str = "dev-only-insecure-secret-change-me"
    jwt_alg: str = "HS256"
    access_ttl_seconds: int = 900          # 15 min
    refresh_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days

    # --- Cookies -------------------------------------------------------------
    cookie_domain: str | None = None
    cookie_secure: bool | None = None      # None -> derive from is_prod
    cookie_samesite: str = "lax"
    access_cookie: str = "volcast_at"
    refresh_cookie: str = "volcast_rt"

    # --- Storage -------------------------------------------------------------
    db_path: str = ".volcast/volcast.db"

    # --- Model / jobs --------------------------------------------------------
    engine: str = "drift"
    allow_chronos: bool = False

    # Demo access: lets a single-user local install see the dashboard without
    # enrolling a passkey first. NEVER honoured when env=prod (see allow_demo
    # property), so enabling it here cannot weaken a deployed instance.
    allow_demo: bool = True

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def demo_enabled(self) -> bool:
        """Demo sign-in is a dev-only convenience; prod always requires a passkey."""
        return bool(self.allow_demo) and not self.is_prod

    @property
    def secure_cookies(self) -> bool:
        """Secure-cookie flag resolved from env unless explicitly overridden."""
        if self.cookie_secure is not None:
            return bool(self.cookie_secure)
        return self.is_prod

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.expected_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
