"""Turn a feasible plan into dated/labelled viewing sessions.

Episodes are the smallest unit, a movie is never split across sessions, a
per-title break (if requested) is placed before the title's first episode,
and no session exceeds its capacity. If the total fits the budget but not the
given daily slots, the schedule is reported as infeasible with what is missing.
"""

from __future__ import annotations

from app.planning.feasibility import PlanEntry


def build_schedule(entries: list[PlanEntry], times: dict, titles: dict[str, str], slots: list[dict],
                   break_minutes: int = 0) -> dict:
    sessions = [{"index": n, "label": s["label"], "capacity_minutes": float(s["minutes"]), "used_minutes": 0.0,
                 "entries": []} for n, s in enumerate(slots)]
    problems: list[str] = []
    unscheduled: list[dict] = []
    cur = 0
    max_cap = max((s["capacity_minutes"] for s in sessions), default=0.0)
    for e in entries:
        t = times[e.item_key]
        if t.is_movie:
            units = [(1, 1, float(t.runtime) + break_minutes)]
        else:
            units = [(ep, ep, float(t.per_episode) + (break_minutes if ep == e.episode_start else 0))
                     for ep in range(e.episode_start, e.episode_end + 1)]
        if t.is_movie and units[0][2] > max_cap:
            problems.append(f"{titles.get(e.item_key, e.item_key)} ({units[0][2]:.0f} min) is longer than any "
                            f"single session; a movie is not split.")
            unscheduled.append({"item_key": e.item_key, "episodes": [1, 1]})
            continue
        for ep_s, ep_e, minutes in units:
            while cur < len(sessions) and sessions[cur]["used_minutes"] + minutes > sessions[cur]["capacity_minutes"] + 1e-9:
                cur += 1
            if cur >= len(sessions):
                unscheduled.append({"item_key": e.item_key, "episodes": [ep_s, ep_e]})
                continue
            s = sessions[cur]
            last = s["entries"][-1] if s["entries"] else None
            if last and last["item_key"] == e.item_key and not t.is_movie and last["episode_end"] == ep_s - 1:
                last["episode_end"] = ep_e
                last["minutes"] = round(last["minutes"] + minutes, 2)
            else:
                s["entries"].append({"item_key": e.item_key, "title": titles.get(e.item_key, e.item_key),
                                     "episode_start": ep_s, "episode_end": ep_e, "minutes": round(minutes, 2),
                                     "kind": "movie" if t.is_movie else "episodes"})
            s["used_minutes"] = round(s["used_minutes"] + minutes, 2)
    # validation: per-title totals and capacities
    for s in sessions:
        if s["used_minutes"] > s["capacity_minutes"] + 1e-6:
            problems.append(f"Session {s['label']} is over capacity.")
    if unscheduled:
        missing = 0.0
        for u in unscheduled:
            t = times[u["item_key"]]
            missing += float(t.runtime) if t.is_movie else (u["episodes"][1] - u["episodes"][0] + 1) * t.per_episode
        problems.append(f"The plan fits the total budget but not the given sessions: about {missing:.0f} more "
                        f"minutes of session time are needed.")
    return {"feasible": not problems, "sessions": sessions, "unscheduled": unscheduled, "problems": problems}
