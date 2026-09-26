"""Export one JSON Schema containing every wire and API model (input for dashboard types)."""

from __future__ import annotations

import json
import sys

from pydantic.json_schema import models_json_schema

from .api import IngestResult, LiveEvent, NodeView, TrackDetail
from .models import Heartbeat, NodeRegistration, Observation, Track
from .traces import FeatureTraceHeader, TraceRequest, TraceUnavailable

EXPORTED = (
    Observation, Heartbeat, Track, NodeRegistration, NodeView, TrackDetail, LiveEvent, IngestResult,
    FeatureTraceHeader, TraceRequest, TraceUnavailable,
)


def export_schema() -> dict:
    _, schema = models_json_schema(
        [(model, "serialization") for model in EXPORTED], title="KuuloSchema"
    )
    # json-schema-to-typescript only emits types reachable from the root,
    # so the root object references every exported model.
    schema["type"] = "object"
    schema["properties"] = {m.__name__: {"$ref": f"#/$defs/{m.__name__}"} for m in EXPORTED}
    schema["additionalProperties"] = False
    return schema


def main() -> None:
    json.dump(export_schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
