"""In-memory circular Episode Buffer for streaming episodes and critique replays."""

from __future__ import annotations
from collections import deque
from rl_ide.contracts.adapter import EpisodeLog
from rl_ide.reward.evaluator import RewardEvaluator


class EpisodeBuffer:
    """Fixed-capacity buffer storing recent EpisodeLog instances."""

    def __init__(self, max_episodes: int = 200):
        self.max_episodes = max_episodes
        self._buffer: deque[EpisodeLog] = deque(maxlen=max_episodes)
        self.locked_ids: set[int] = set()

    def lock_episode(self, episode_id: int) -> None:
        """Locks an episode by ID so it cannot be degraded or pruned."""
        self.locked_ids.add(episode_id)

    def unlock_episode(self, episode_id: int) -> None:
        """Unlocks a locked episode ID."""
        self.locked_ids.discard(episode_id)

    def append(self, episode: EpisodeLog) -> None:
        """Appends a completed episode log."""
        self._buffer.append(episode)

    def get_recent(self, n: int = 20) -> list[EpisodeLog]:
        """Returns the most recent n episodes, newest last."""
        items = list(self._buffer)
        return items[-n:] if len(items) >= n else items

    def get_by_id(self, episode_id: int) -> EpisodeLog | None:
        """Finds an episode by its ID."""
        for ep in self._buffer:
            if ep.episode_id == episode_id:
                return ep
        return None

    def all(self) -> list[EpisodeLog]:
        """Returns all cached episodes."""
        return list(self._buffer)

    def rescore(self, evaluator: RewardEvaluator) -> list[EpisodeLog]:
        """Retroactively re-scores all buffered episodes with a new RewardEvaluator."""
        episodes = list(self._buffer)
        evaluator.rescore_episodes(episodes)
        return episodes

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
