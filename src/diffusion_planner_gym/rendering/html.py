"""Fill the HTML template with an episode recording."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

TEMPLATE_PATH = Path(__file__).with_name("template.html")


def render_episode_html(
    snapshots: list[dict[str, Any]], *, title: str, dt: float
) -> str:
    """A standalone HTML page that replays `snapshots` (see `snapshot.py`)."""
    template = TEMPLATE_PATH.read_text()
    data = json.dumps(
        {"title": title, "dt": dt, "frames": snapshots}, separators=(",", ":")
    )
    # Keep the JSON from closing its <script> element.
    data = data.replace("</", "<\\/")
    return template.replace("__TITLE__", html.escape(title)).replace(
        "__EPISODE_DATA__", data
    )
