"""Write the API's OpenAPI schema to a file.

The frontend's TypeScript types are generated from this, so the wire contract
has one source: the Pydantic models. Hand-written interfaces on the client
drift from the server silently, and the first symptom is a field that is
`undefined` in production and fine in every test.

Runs offline -- building the app does not open a database connection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.main import create_app

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "frontend" / "openapi.json"


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    schema = create_app().openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    # `sort_keys` and a trailing newline so the file is byte-stable across
    # machines; CI compares it against a fresh export to catch drift.
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
