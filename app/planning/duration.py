"""Viewing-time facts per title, with their evidence.

    remaining_episodes = known_total_episodes - watched_episodes
    full_cost          = remaining_episodes * minutes_per_episode     (TV / ONA / OVA / Special)
    movie_cost         = total runtime                                 (a movie is one indivisible unit)

Durations come from the catalogue's Duration field, which is a per-episode
average ("24 min per ep"); plans say so instead of claiming minute-exact totals.
Unknown episode counts or durations make a title unplannable (never 12 x 24).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TitleTime:
    idx: int
    is_movie: bool
    total_episodes: int | None
    watched: int
    per_episode: float | None
    runtime: float | None
    basis: str | None

    @property
    def remaining(self) -> int | None:
        if self.total_episodes is None:
            return None
        return max(0, self.total_episodes - self.watched)

    @property
    def known(self) -> bool:
        if self.is_movie:
            return self.runtime is not None
        return self.total_episodes is not None and self.per_episode is not None

    def full_cost(self) -> float | None:
        if not self.known:
            return None
        if self.is_movie:
            return float(self.runtime) if self.watched == 0 else 0.0
        return float(self.remaining * self.per_episode)

    def partial_cost(self, n_episodes: int) -> float | None:
        if not self.known or self.is_movie:
            return self.full_cost() if self.is_movie and n_episodes else None
        return float(n_episodes * self.per_episode)

    def evidence(self) -> dict:
        return {"per_episode_minutes": self.per_episode, "total_episodes": self.total_episodes,
                "watched_episodes": self.watched, "runtime_minutes": self.runtime,
                "basis": self.basis or "unknown",
                "note": "per-episode average from the catalogue Duration field" if self.basis ==
                "per_episode_average" else "catalogue runtime"}


def title_time(snap, i: int, watched: int = 0) -> TitleTime:
    eps = snap.episodes[i]
    per = snap.duration[i]
    tot = snap.total_minutes[i]
    is_movie = (snap.type[i] == "Movie") or (np.isfinite(eps) and int(eps) == 1)
    basis = "total_runtime" if is_movie else "per_episode_average"
    return TitleTime(idx=i, is_movie=bool(is_movie), total_episodes=None if not np.isfinite(eps) else int(eps),
                     watched=int(watched), per_episode=None if not np.isfinite(per) else float(per),
                     runtime=(float(tot) if np.isfinite(tot) else (float(per) if is_movie and np.isfinite(per) else None)),
                     basis=basis)
