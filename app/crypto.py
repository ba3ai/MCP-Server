import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv
load_dotenv()

def _get_fernet() -> Fernet:
    key = os.environ.get("FERNET_KEY")
    if not key:
        raise RuntimeError("FERNET_KEY is not set")
    if isinstance(key, str):
        key = key.encode("utf-8")
    return Fernet(key)

def encrypt(value: str) -> str:
    f = _get_fernet()
    return f.encrypt(value.encode("utf-8")).decode("utf-8")

def decrypt(value: str) -> str:
    f = _get_fernet()
    return f.decrypt(value.encode("utf-8")).decode("utf-8")
