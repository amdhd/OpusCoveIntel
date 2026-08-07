"""Password hashing on `hashlib.scrypt`, from the standard library.

No new dependency, and not a compromise: scrypt is memory-hard, it is what
`hashlib` ships, and the parameters are stored beside each hash so they can be
raised later without invalidating existing passwords.

The encoded form is a single string, `scrypt$n$r$p$<salt_b64>$<hash_b64>`, so
one column holds everything needed to verify -- there is no second column to
forget to migrate when the cost parameters change.

Two properties this module must not lose:

* **Comparison is constant-time.** `secrets.compare_digest`, never `==`; a
  byte-by-byte comparison leaks how much of a hash matched.
* **Verification never raises on malformed input.** A corrupt or truncated hash
  in the database must read as "wrong password", not as a 500 that tells an
  attacker they found something interesting.
"""

from __future__ import annotations

import base64
import hashlib
import secrets

# OWASP's floor for scrypt at the time of writing (n=2^17, r=8, p=1). ~128 MB
# per hash, which is the point -- it is what makes offline cracking expensive.
# Raising these is safe: existing hashes carry their own parameters.
DEFAULT_N = 1 << 17
DEFAULT_R = 8
DEFAULT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32

_SCHEME = "scrypt"


def _maxmem(n: int, r: int) -> int:
    """Memory ceiling to hand `hashlib.scrypt`.

    It allocates roughly `128 * n * r` bytes and refuses to exceed `maxmem`,
    whose default (32 MB) is below what n=2^17 needs -- so leaving it unset
    makes every hash raise. Two times the nominal requirement, for headroom.
    """
    return 128 * n * r * 2


def hash_password(
    password: str, *, n: int = DEFAULT_N, r: int = DEFAULT_R, p: int = DEFAULT_P
) -> str:
    """Hash a password with a fresh random salt."""
    if not password:
        raise ValueError("refusing to hash an empty password")

    salt = secrets.token_bytes(SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=KEY_BYTES,
        maxmem=_maxmem(n, r),
    )
    return "$".join(
        (
            _SCHEME,
            str(n),
            str(r),
            str(p),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """Check a password against an encoded hash.

    Returns False for anything it cannot parse. A malformed row is a failed
    login, not an exception -- see the module docstring.
    """
    try:
        scheme, n_raw, r_raw, p_raw, salt_b64, hash_b64 = encoded.split("$")
        if scheme != _SCHEME:
            return False
        n, r, p = int(n_raw), int(r_raw), int(p_raw)
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(hash_b64, validate=True)
    except (ValueError, TypeError):
        return False

    if not password or not salt or not expected:
        return False

    try:
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=_maxmem(n, r),
        )
    except ValueError:
        # Parameters that scrypt rejects (n not a power of two, absurd cost).
        return False

    return secrets.compare_digest(candidate, expected)


def needs_rehash(
    encoded: str, *, n: int = DEFAULT_N, r: int = DEFAULT_R, p: int = DEFAULT_P
) -> bool:
    """True when a stored hash is weaker than the current parameters.

    Lets a login path upgrade a hash transparently once the cost is raised.
    """
    try:
        scheme, n_raw, r_raw, p_raw, _, _ = encoded.split("$")
    except ValueError:
        return True
    if scheme != _SCHEME:
        return True
    try:
        return (int(n_raw), int(r_raw), int(p_raw)) < (n, r, p)
    except ValueError:
        return True


def new_session_token() -> str:
    """A high-entropy opaque session token.

    32 bytes, URL-safe. This is the only secret that ever reaches the client;
    the database stores its SHA-256 (see `app.auth.service`), so a leaked
    database dump does not hand over live sessions.
    """
    return secrets.token_urlsafe(32)


def token_fingerprint(token: str) -> str:
    """SHA-256 of a session token, hex-encoded.

    Plain SHA-256 rather than scrypt, deliberately: the token is 256 bits of
    random already, so there is nothing to brute-force and a slow KDF would
    only add latency to every authenticated request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
