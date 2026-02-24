import os
import httpx
from jose import jwt
from jose.exceptions import JWTError
from dotenv import load_dotenv
load_dotenv()

ISSUER_DOMAIN = os.environ.get("OAUTH_ISSUER_DOMAIN")
AUDIENCE = os.environ.get("OAUTH_AUDIENCE")
ISSUER = f"https://{ISSUER_DOMAIN}/" if ISSUER_DOMAIN else None
ALGORITHMS = os.environ.get("OAUTH_ALGORITHMS", "RS256").split(",")

_jwks_cache = None

async def _get_jwks():
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    if not ISSUER:
        raise RuntimeError("OAUTH_ISSUER_DOMAIN is not set")
    url = f"{ISSUER}.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        _jwks_cache = r.json()
        return _jwks_cache

async def verify_bearer_token(auth_header: str) -> dict:
    if not AUDIENCE:
        raise RuntimeError("OAUTH_AUDIENCE is not set")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise PermissionError("Missing Bearer token")

    token = auth_header.split(" ", 1)[1].strip()
    jwks = await _get_jwks()

    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        key = next(k for k in jwks["keys"] if k.get("kid") == kid)

        claims = jwt.decode(
            token,
            key,
            algorithms=ALGORITHMS,
            audience=AUDIENCE,
            issuer=ISSUER,
        )
        return claims
    except (StopIteration, JWTError) as e:
        raise PermissionError(f"Invalid token: {e}")
