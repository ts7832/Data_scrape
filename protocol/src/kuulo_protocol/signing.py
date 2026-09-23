"""Ed25519 signing over canonical JSON. Every node message is signed with the node's key."""

from __future__ import annotations

import base64
import binascii
import json

from nacl.exceptions import BadSignatureError, CryptoError
from nacl.signing import SigningKey, VerifyKey
from pydantic import BaseModel


def canonical_bytes(msg: BaseModel) -> bytes:
    """Deterministic bytes of a message, excluding its signature field."""
    data = msg.model_dump(mode="json", exclude={"signature"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text, validate=True)


def _pair(key: SigningKey) -> tuple[str, str]:
    return _b64(bytes(key)), _b64(bytes(key.verify_key))


def generate_keypair() -> tuple[str, str]:
    """Return (private_key_b64, public_key_b64)."""
    return _pair(SigningKey.generate())


def keypair_from_seed(seed: bytes) -> tuple[str, str]:
    """Deterministic keypair from a 32-byte seed (used by the simulator)."""
    return _pair(SigningKey(seed))


def sign[M: BaseModel](msg: M, private_key_b64: str) -> M:
    signature = SigningKey(_unb64(private_key_b64)).sign(canonical_bytes(msg)).signature
    return msg.model_copy(update={"signature": _b64(signature)})


def verify(msg: BaseModel, public_key_b64: str) -> bool:
    """True only for a valid signature by this key. Never raises on bad input."""
    signature = getattr(msg, "signature", "")
    if not signature:
        return False
    try:
        VerifyKey(_unb64(public_key_b64)).verify(canonical_bytes(msg), _unb64(signature))
    except (BadSignatureError, CryptoError, binascii.Error, ValueError, TypeError):
        return False
    return True
