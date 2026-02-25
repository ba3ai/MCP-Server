from contextvars import ContextVar
from typing import Optional, Dict, Any

# Per-request user identity extracted from MCP OAuth bearer token.
current_user: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_user", default=None)
