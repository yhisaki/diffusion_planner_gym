"""Fill the HTML template with an episode recording."""

from __future__ import annotations

import base64
import gzip
import html
import json
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).with_name("template.html")


def render_episode_html(
    snapshots: list[dict[str, Any]], *, title: str, dt: float
) -> str:
    """A standalone HTML page that replays `snapshots` (see `snapshot.py`).

    The episode JSON is embedded gzip-compressed and base64-encoded; the page inflates
    it with the browser's `DecompressionStream`.
    """
    template = TEMPLATE_PATH.read_text()
    data = json.dumps(
        {"title": title, "dt": dt, "frames": snapshots}, separators=(",", ":")
    )
    # mtime=0 keeps the output deterministic. Base64 cannot close the <script> element.
    payload = base64.b64encode(gzip.compress(data.encode(), compresslevel=6, mtime=0))
    return template.replace("__TITLE__", html.escape(title)).replace(
        "__EPISODE_DATA__", payload.decode("ascii")
    )
