"""Authentication service state: session store and public path configuration.

Per D-10: PUBLIC_PATHS includes status, settings/app, login, setup, plus /api/health (RESEARCH Pitfall 8).
"""

COOKIE_NAME = "paperless_ai_session"
SALT_SIZE = 16
SCRYPT_N = 2**14   # OWASP minimum (CONTEXT.md "Claude's Discretion"; RESEARCH.md A1)
SCRYPT_R = 8
SCRYPT_P = 1

# Module-level session store — lives here so middleware can read without DI context.
# Keyed by token (secrets.token_hex(32)). Value is literal "authenticated" sentinel.
SESSIONS: dict[str, str] = {}

# Public paths that bypass auth. Per CONTEXT.md D-10 plus /api/health (RESEARCH Pitfall 8).
PUBLIC_PATHS: set[tuple[str, str]] = {
    ("GET", "/api/auth/status"),
    ("GET", "/api/settings/app"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/setup"),
    ("GET", "/api/health"),
}