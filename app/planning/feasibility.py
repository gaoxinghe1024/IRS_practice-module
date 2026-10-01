"""Hard feasibility for viewing plans (budget, title cap, prerequisites, episode ranges, order).

Every plan returned by the optimiser or produced by a user edit is re-checked
here; a plan that fails is never labelled as fitting the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanEntry:
    idx: int
    item_key: str
    episode_start: int          # 1-based, inclusive (movies: 1)
    episode_end: int            # inclusive
    minutes: float
    is_full: bool
    prerequisite_of: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


@dataclass
class Feasibility:
    feasible: bool
    problems: list[str]
    total_minutes: float
    budget_minutes: float
    remaining_minutes: float
    titles: int
    max_titles: int

    def public(self) -> dict:
        return {"feasible": self.feasible, "problems": self.problems, "total_minutes": round(self.total_minutes, 1),
                "budget_minutes": self.budget_minutes, "remaining_minutes": round(self.remaining_minutes, 1),
                "titles": self.titles, "max_titles": self.max_titles}


def check(entries: list[PlanEntry], times: dict, budget: float, max_titles: int, break_minutes: int,
          completed_keys: set[str], mode: str) -> Feasibility:
    problems: list[str] = []
    keys = [e.item_key for e in entries]
    if len(set(keys)) != len(keys):
        problems.append("A title appears more than once.")
    total = 0.0
    position = {k: n for n, k in enumerate(keys)}
    for e in entries:
        t = times.get(e.item_key)
        if t is None or not t.known:
            problems.append(f"{e.item_key}: duration or episode count unknown.")
            continue
        first = t.watched + 1
        last = t.total_episodes if not t.is_movie else 1
        if t.is_movie:
            if (e.episode_start, e.episode_end) != (1, 1):
                problems.append(f"{e.item_key}: a movie cannot be split.")
            expected = t.runtime
        else:
            if e.episode_start != first:
                problems.append(f"{e.item_key}: must start after your progress (episode {first}).")
            if e.episode_end > last or e.episode_end < e.episode_start:
                problems.append(f"{e.item_key}: episode range {e.episode_start}-{e.episode_end} is invalid.")
            if mode == "complete" and e.episode_end != last:
                problems.append(f"{e.item_key}: complete mode requires finishing the title.")
            expected = (e.episode_end - e.episode_start + 1) * t.per_episode
        if abs(e.minutes - expected) > 0.01:
            problems.append(f"{e.item_key}: minutes do not match the episode range.")
        total += expected + break_minutes
        for req in e.requires:
            if req in completed_keys:
                continue
            if req not in position:
                problems.append(f"{e.item_key}: prerequisite {req} is neither watched nor in the plan.")
            elif position[req] > position[e.item_key]:
                problems.append(f"{e.item_key}: prerequisite {req} must come first.")
            else:
                dep = entries[position[req]]
                t_req = times.get(req)
                if t_req is not None and not t_req.is_movie and dep.episode_end != t_req.total_episodes:
                    problems.append(f"{e.item_key}: prerequisite {req} must be watched in full first.")
    if total > budget + 1e-6:
        problems.append(f"Total {total:.0f} min exceeds the budget of {budget:.0f} min.")
    if len(entries) > max_titles:
        problems.append(f"{len(entries)} titles exceed the maximum of {max_titles}.")
    return Feasibility(not problems, problems, total, budget, budget - total, len(entries), max_titles)
