"""Plan exports: JSON, Markdown and printable HTML (no secrets, paths or debug logs)."""

from __future__ import annotations

import html
import json


def to_json(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2, default=str)


def _range(it: dict) -> str:
    if it.get("type") == "Movie" or it["episode_start"] == it["episode_end"] == 1 and it.get("is_full"):
        return "Full movie" if it.get("type") == "Movie" else "Episode 1"
    return f"Episodes {it['episode_start']}-{it['episode_end']}" + ("" if it.get("is_full") else " (partial)")


def _method(plan: dict) -> str:
    if plan.get("algorithm") == "manual_edit":
        return "edited by you (feasibility re-checked after the edit)"
    return str((plan.get("objective") or {}).get("optimizer") or plan.get("algorithm") or "unknown")


def _span(x: dict) -> str:
    return "movie" if x["kind"] == "movie" else f"ep {x['episode_start']}-{x['episode_end']}"


def to_markdown(plan: dict) -> str:
    lines = [f"# {plan.get('title') or 'Viewing plan'}", ""]
    lines.append(f"- Budget: {plan['budget_minutes']} min; planned: {plan['total_minutes']:.0f} min; "
                 f"remaining: {plan['remaining_minutes']:.0f} min")
    lines.append(f"- Titles: {plan['titles']} (max {plan['max_titles']}); mode: {plan['mode']}")
    lines.append(f"- Budget check: {'OK' if plan['feasibility']['feasible'] else 'NOT satisfied'}")
    lines += ["", "| # | Title | Range | Minutes | Why | Watch first |", "|---|---|---|---|---|---|"]
    for n, it in enumerate(plan["items"], start=1):
        why = "; ".join(r["text"] for r in it.get("reasons", [])[:2]).replace("|", "/")
        first = ", ".join(it.get("requires_titles", [])) or "-"
        lines.append(f"| {n} | {it['title'].replace('|', '/')} | {_range(it)} | {it['minutes']:.0f} | {why} | {first} |")
    sched = plan.get("schedule")
    if sched and sched.get("sessions"):
        lines += ["", "## Sessions", ""]
        for s in sched["sessions"]:
            parts = []
            for x in s["entries"]:
                parts.append(f"{x['title']} ({_span(x)})")
            lines.append(f"- **{s['label']}** ({s['used_minutes']:.0f}/{s['capacity_minutes']:.0f} min): "
                         + (", ".join(parts) or "free"))
        if not sched["feasible"]:
            lines.append("")
            lines += [f"> {p}" for p in sched["problems"]]
    lines += ["", "## Notes", "",
              "- Minutes use the catalogue's per-episode average duration, not exact runtimes.",
              f"- Dataset: {plan.get('dataset_mode')} ({plan.get('dataset_version')}); plan method: {_method(plan)}."]
    return "\n".join(lines) + "\n"


def to_html(plan: dict) -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{n}</td><td>{e(it['title'])}</td><td>{e(_range(it))}</td><td>{it['minutes']:.0f}</td>"
        f"<td>{e('; '.join(r['text'] for r in it.get('reasons', [])[:2]))}</td>"
        f"<td>{e(', '.join(it.get('requires_titles', [])) or '-')}</td></tr>"
        for n, it in enumerate(plan["items"], start=1))
    sched = ""
    if plan.get("schedule") and plan["schedule"].get("sessions"):
        items = "".join(
            f"<li><strong>{e(s['label'])}</strong> ({s['used_minutes']:.0f}/{s['capacity_minutes']:.0f} min): "
            + e(", ".join(f"{x['title']} ({_span(x)})" for x in s["entries"]) or "free") + "</li>"
            for s in plan["schedule"]["sessions"])
        sched = f"<h2>Sessions</h2><ul>{items}</ul>"
    status = "OK" if plan["feasibility"]["feasible"] else "NOT satisfied"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(plan.get('title') or 'Viewing plan')}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f0f0f0}}@media print{{body{{margin:0}}}}</style></head><body>
<h1>{e(plan.get('title') or 'Viewing plan')}</h1>
<p>Budget {plan['budget_minutes']} min &middot; planned {plan['total_minutes']:.0f} min &middot; remaining
{plan['remaining_minutes']:.0f} min &middot; {plan['titles']} titles (max {plan['max_titles']}) &middot; budget check: {status}</p>
<table><thead><tr><th>#</th><th>Title</th><th>Range</th><th>Minutes</th><th>Why</th><th>Watch first</th></tr></thead>
<tbody>{rows}</tbody></table>{sched}
<p><small>Minutes use the catalogue's per-episode average duration. Dataset {e(str(plan.get('dataset_mode')))}
({e(str(plan.get('dataset_version')))}); plan method: {e(_method(plan))}.</small></p></body></html>"""
