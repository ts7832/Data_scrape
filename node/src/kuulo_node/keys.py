"""The node's Ed25519 identity: created on first run, then reused (the server pins it)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from kuulo_protocol.signing import generate_keypair


@dataclass(frozen=True)
class NodeKeys:
    private_key: str
    public_key: str


def load_or_create_keys(path: Path) -> NodeKeys:
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        return NodeKeys(private_key=data["private_key"], public_key=data["public_key"])
    private_key, public_key = generate_keypair()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"private_key": private_key, "public_key": public_key}))
    os.chmod(path, 0o600)
    return NodeKeys(private_key=private_key, public_key=public_key)
