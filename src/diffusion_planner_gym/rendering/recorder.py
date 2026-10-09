"""Save each episode of an environment as a standalone HTML replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym


class RecordEpisodeHtml(gym.Wrapper):
    """Write `<output_dir>/<name_prefix>-<episode>.html` when an episode ends.

    The wrapped environment must be created with `render_mode="html"`.
    """

    def __init__(
        self, env: gym.Env, output_dir: str | Path, name_prefix: str = "episode"
    ) -> None:
        super().__init__(env)
        if env.render_mode != "html":
            raise ValueError('RecordEpisodeHtml needs render_mode="html"')
        self.output_dir = Path(output_dir)
        self.name_prefix = name_prefix
        self.episode_id = 0
        self._recording = False

    def reset(self, **kwargs: Any) -> Any:
        if self._recording:
            self._save()
        result = self.env.reset(**kwargs)
        self._recording = True
        return result

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            self._save()
        return result

    def close(self) -> None:
        if self._recording:
            self._save()
        super().close()

    def _save(self) -> None:
        html = self.env.render()
        assert isinstance(html, str)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.name_prefix}-{self.episode_id:03d}.html"
        path.write_text(html)
        self.episode_id += 1
        self._recording = False
