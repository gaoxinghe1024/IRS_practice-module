"""Parse MAL's public recommendation cards, never unrelated anime/sidebar links."""
import html
import re


def parse_recommendations(page, target_id):
    if not re.search(r"<h2[^>]*>Recommendations</h2>", page):
        raise ValueError("Not a MAL recommendation page")
    entries = []
    for block in re.split(r'<div class="borderClass">\s*<table\b', page)[1:]:
        pair = re.search(r'href="/recommendations/anime/(\d+)-(\d+)"[^>]*title="Permalink"', block)
        if not pair:
            continue
        endpoints = [int(value) for value in pair.groups()]
        if target_id not in endpoints or endpoints[0] == endpoints[1]:
            raise ValueError("Recommendation permalink belongs to a different target")
        aid = endpoints[1] if endpoints[0] == target_id else endpoints[0]
        title = re.search(r'<a href="https://myanimelist.net/anime/' + str(aid) + r'/[^\"]*"><strong>(.*?)</strong>', block, re.S)
        extra = re.search(r'Read recommendations by\s*<strong>([\d,]+)</strong>\s*more users?', block)
        listed = len(re.findall(r'Recommended by\s*<a\b', block))
        if listed < 1:
            raise ValueError("Missing recommendation author count; do not invent votes")
        votes = 1 + int(extra[1].replace(",", "")) if extra else listed
        # MAL normally embeds all hidden recommendations; require visible evidence for counts.
        if extra and listed != votes:
            raise ValueError(f"Vote count mismatch for {target_id}-{aid}: {votes} versus {listed}")
        entries.append(dict(entry=dict(mal_id=aid, title=html.unescape(title[1]) if title else str(aid)),
                            votes=votes, url=f"https://myanimelist.net/recommendations/anime/{endpoints[0]}-{endpoints[1]}"))
    if not entries and not re.search(r"No recommendations|No recommendations have been|No recommendations yet", page, re.I):
        raise ValueError("No cards found and no explicit empty-page message")
    return {"data": entries}
