"""Seeded synthetic world for running and verifying the whole system without Kaggle files.

Everything here is fictional and labelled ``synthetic`` (item keys
``synthetic:<seed>:<n>``, mal_id NULL, provenance ``synthetic``). Generic genre
and theme words are used as vocabulary, but titles, studios, characters,
synopses and scenes are invented. Evaluation files are written separately from
the corpus; they validate implementation, never recommendation quality.

Latent structure (so CF / MF / NCF have something real to learn):
  * items: genres/themes/demographic/studios + hidden latent vector h_i + quality q_i
  * users: tag preference weights + hidden latent u_h + rating bias + activity level
  * watch choice ~ popularity^a * exp(b * affinity); rating = bias + affinity + quality + noise
  * collections are generated from a separate "collection theme" process, not from ratings
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.ids import synthetic_key
from pipelines.common import (
    PIPELINE_VERSION,
    dataset_version,
    dependency_versions,
    sha256_file,
    sha256_json,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from pipelines.processed_io import ProcessedWriter

GENRES = ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
          "Slice of Life", "Sports", "Supernatural", "Suspense"]
THEMES = ["Psychological", "Time Travel", "School", "Mecha", "Military", "Music", "Isekai", "Detective",
          "Mythology", "Space", "Survival", "Workplace", "Historical", "Martial Arts", "Iyashikei", "Super Power",
          "Strategy Game", "Gourmet"]
DEMOGRAPHICS = ["Shounen", "Seinen", "Shoujo", "Josei", "Kids"]
STUDIOS = ["Studio Aster", "Kumo Works", "Lantern Frame", "Brightwell Animation", "Hoshikage Studio",
           "Paper Crane Pictures", "Nightjar Motion", "Tidewater Studio", "Orbit Line Animation", "Maple Arc"]
SOURCES = ["Original", "Manga", "Light novel", "Game", "Novel", "Web manga"]
TYPES = [("TV", 0.55), ("Movie", 0.14), ("OVA", 0.1), ("ONA", 0.13), ("Special", 0.08)]
RATINGS_CR = [("PG-13 - Teens 13 or older", 0.5), ("G - All Ages", 0.18), ("PG - Children", 0.12),
              ("R - 17+ (violence & profanity)", 0.15), ("R+ - Mild Nudity", 0.05)]
ADJ = ["Crimson", "Silent", "Hollow", "Azure", "Last", "Paper", "Iron", "Wandering", "Ninth", "Glass",
       "Midnight", "Distant", "Clockwork", "Velvet", "Amber", "Frozen", "Seventh", "Quiet", "Burning", "Lunar",
       "Copper", "Drifting", "Violet", "Northern", "Sunken", "Hidden", "Gentle", "Restless", "Starlit", "Rusted"]
NOUN = ["Orbit", "Tide", "Garden", "Requiem", "Signal", "Harbor", "Frontier", "Archive", "Covenant", "Compass",
        "Ember", "Atlas", "Sonata", "Circuit", "Lantern", "Meridian", "Parade", "Kingdom", "Voyage", "Horizon",
        "Relay", "Canvas", "Chorus", "Ledger", "Beacon", "Labyrinth", "Carnival", "Observatory", "Station", "Echo"]
FIRST = ["Aori", "Kenta", "Mirei", "Sora", "Haruki", "Yuzu", "Rin", "Tomo", "Nao", "Kaede", "Itsuki", "Hina",
         "Riku", "Saya", "Daichi", "Emi", "Kou", "Mana", "Shun", "Toa"]
LAST = ["Kirishima", "Aozora", "Tsukimura", "Hayase", "Mikado", "Shirogane", "Amanogawa", "Kuroda", "Sakuraba",
        "Minase", "Tachibana", "Hoshino", "Asakura", "Fujisaki", "Nanase"]
# Plot motifs with paraphrases: synopses use the first phrasing, evaluation queries another.
MOTIFS = [
    ("loses their memory after a train accident", "can't remember who they are after a crash on the railway"),
    ("inherits a lighthouse on a remote island", "gets a lighthouse on a faraway island from a relative"),
    ("builds a robot out of scrap metal", "makes a machine friend from junkyard parts"),
    ("joins a failing school orchestra", "tries to save a music club at school that is about to close"),
    ("is chased by a masked hunter", "keeps running from a hunter wearing a mask"),
    ("opens a small ramen shop", "runs a tiny noodle restaurant"),
    ("wakes up in a game world", "finds themselves inside a video game"),
    ("investigates disappearances in a mountain village", "looks into people vanishing in a village in the hills"),
    ("pilots a giant mech against invaders", "drives a huge robot to fight off an invasion"),
    ("travels back to prevent a friend's death", "goes back in time to stop a friend from dying"),
    ("trains for a national relay race", "prepares for a big running competition as a team"),
    ("guards a library of forbidden books", "protects a collection of banned books"),
    ("receives letters from the future", "gets messages sent from years ahead"),
    ("survives on a drifting space station", "tries to stay alive on a broken station floating in space"),
    ("befriends a shy fox spirit", "becomes friends with a timid fox ghost"),
    ("plays high-stakes card games", "gambles in dangerous card matches"),
]
CONTENT_EXPLICIT_COUNT = 3  # fictional placeholder entries that exercise the explicit-content rule


def _choice_p(rng: np.random.Generator, pairs: list[tuple[str, float]]) -> str:
    names, probs = zip(*pairs, strict=True)
    p = np.array(probs) / sum(probs)
    return str(rng.choice(names, p=p))


def generate(settings: Settings, log=print) -> dict[str, Any]:
    cfg = settings.synthetic
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    n_anime = int(cfg.get("n_anime", 200))
    n_users = int(cfg.get("n_users", 800))
    target = int(cfg.get("target_ratings", 20000))
    out = settings.synthetic_dir
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ items
    tag_vocab = [("genre", g) for g in GENRES] + [("theme", t) for t in THEMES] + [
        ("demographic", d) for d in DEMOGRAPHICS]
    tag_index = {t: i for i, t in enumerate(tag_vocab)}
    d_hidden = int(cfg.get("n_latent_hidden", 4))
    items: list[dict] = []
    titles_used: set[str] = set()
    franchises: list[list[int]] = []
    n = 0

    def new_title() -> str:
        for _ in range(200):
            pat = rng.integers(0, 3)
            a, b, c = rng.choice(ADJ), rng.choice(NOUN), rng.choice(NOUN)
            t = (f"The {a} {b}" if pat == 0 else f"{b} of the {a} {c}" if pat == 1 else f"{a} {b}")
            if t not in titles_used:
                titles_used.add(t)
                return t
        raise RuntimeError("title space exhausted")

    base_count = n_anime - CONTENT_EXPLICIT_COUNT
    while n < base_count:
        n += 1
        genres = list(rng.choice(GENRES, size=int(rng.integers(1, 4)), replace=False))
        themes = list(rng.choice(THEMES, size=int(rng.integers(0, 3)), replace=False))
        demo = [str(rng.choice(DEMOGRAPHICS))] if rng.random() > 0.2 else []
        studios = list(rng.choice(STUDIOS, size=1 if rng.random() > 0.12 else 2, replace=False)) \
            if rng.random() > 0.25 else []
        typ = _choice_p(rng, TYPES)
        if typ == "Movie":
            episodes = 1
            minutes = float(rng.integers(70, 130))
        elif typ == "TV":
            episodes = int(rng.choice([12, 12, 13, 24, 25, 26, 48, 50]))
            minutes = float(rng.choice([23, 24, 24, 24, 25]))
        elif typ == "ONA":
            episodes = int(rng.choice([6, 10, 12, 13, 24]))
            minutes = float(rng.choice([12, 15, 20, 24]))
        elif typ == "OVA":
            episodes = int(rng.choice([2, 3, 4, 6]))
            minutes = float(rng.choice([25, 28, 30, 45]))
        else:
            episodes = int(rng.choice([1, 2, 3]))
            minutes = float(rng.choice([10, 15, 24]))
        status = "Finished Airing" if rng.random() > 0.08 else _choice_p(
            rng, [("Currently Airing", 0.7), ("Not yet aired", 0.3)])
        # edge cases: unknown episodes / duration / score / studios
        if status != "Finished Airing" and rng.random() < 0.6:
            episodes = None
        if rng.random() < 0.05:
            minutes = None
        q = float(rng.normal())
        score = round(float(np.clip(6.9 + 0.65 * q + rng.normal(0, 0.25), 3.5, 9.2)), 2)
        if rng.random() < 0.14 or status == "Not yet aired":
            score = None
        year = int(rng.integers(1995, 2024))
        motifs = [int(m) for m in rng.choice(len(MOTIFS), size=int(rng.integers(1, 3)), replace=False)]
        items.append(dict(n=n, title=new_title(), genres=genres, themes=themes, demographic=demo,
                          studios=studios, type=typ, episodes=episodes, minutes=minutes, status=status,
                          quality=q, score=score, year=year, motifs=motifs,
                          hidden=rng.normal(size=d_hidden).tolist(), source=str(rng.choice(SOURCES)),
                          content_rating=_choice_p(rng, RATINGS_CR), franchise=None, explicit=False))

    # franchises: turn ~15% of TV titles into 2-3 part chains (sequels share tags / latent)
    tv_idx = [i for i, it in enumerate(items) if it["type"] == "TV" and it["status"] == "Finished Airing"]
    rng.shuffle(tv_idx)
    for base in tv_idx[: max(2, len(tv_idx) // 7)]:
        if items[base]["franchise"] is not None:
            continue
        chain = [base]
        for part in range(int(rng.integers(1, 3))):
            n += 1
            src = items[base]
            seq = dict(src)
            seq.update(n=n, title=f"{src['title']} {['Season 2', 'Season 3'][part] if rng.random() > 0.3 else ['Part II', 'Part III'][part]}",
                       episodes=int(rng.choice([12, 13, 24])),
                       year=min(2024, src["year"] + 2 + part), quality=src["quality"] + float(rng.normal(0, 0.3)),
                       hidden=(np.array(src["hidden"]) + rng.normal(0, 0.2, d_hidden)).tolist(),
                       motifs=list(src["motifs"]))
            seq["score"] = None if src["score"] is None else round(float(np.clip(src["score"] + rng.normal(0, 0.2), 3.5, 9.3)), 2)
            items.append(seq)
            chain.append(len(items) - 1)
        for i in chain:
            items[i]["franchise"] = chain
        franchises.append(chain)
    for _ in range(CONTENT_EXPLICIT_COUNT):
        n += 1
        items.append(dict(n=n, title=new_title(), genres=["Erotica"], themes=[], demographic=[], studios=[],
                          type="OVA", episodes=2, minutes=25.0, status="Finished Airing", quality=0.0, score=6.0,
                          year=2010, motifs=[], hidden=[0.0] * d_hidden, source="Original",
                          content_rating="Rx - Hentai", franchise=None, explicit=True))
    n_items = len(items)
    log(f"[synthetic] items={n_items} franchises={len(franchises)}")

    # item feature matrix for the latent model
    X = np.zeros((n_items, len(tag_vocab)))
    for i, it in enumerate(items):
        for g in it["genres"]:
            if ("genre", g) in tag_index:
                X[i, tag_index[("genre", g)]] = 1
        for t in it["themes"]:
            X[i, tag_index[("theme", t)]] = 1
        for d in it["demographic"]:
            X[i, tag_index[("demographic", d)]] = 1
    Xn = X / np.maximum(1, X.sum(1, keepdims=True)) ** 0.5
    H = np.array([it["hidden"] for it in items])
    Q = np.array([it["quality"] for it in items])
    pop = np.exp(1.1 * Q + rng.normal(0, 0.8, n_items))
    pop[[i for i, it in enumerate(items) if it["explicit"]]] *= 0.3
    members = np.round(pop / pop.max() * 250000 + rng.integers(50, 400, n_items)).astype(int)

    # ------------------------------------------------------------------ users + ratings
    served = np.array([not it["explicit"] for it in items])
    W = np.zeros((n_users, len(tag_vocab)))
    for u in range(n_users):
        favs = rng.choice(len(tag_vocab), size=int(rng.integers(3, 6)), replace=False)
        W[u, favs] = rng.gamma(2.0, 1.0, size=len(favs))
        dislikes = rng.choice(len(tag_vocab), size=int(rng.integers(0, 3)), replace=False)
        W[u, dislikes] -= rng.gamma(1.5, 0.8, size=len(dislikes))
    UH = rng.normal(size=(n_users, d_hidden))
    bias = rng.normal(7.0, 0.7, n_users)
    affinity = (W @ Xn.T) / np.sqrt(np.maximum(1e-9, (W ** 2).sum(1, keepdims=True))) + \
        0.45 * (UH @ H.T) / math.sqrt(d_hidden)
    activity = rng.lognormal(3.0, 0.8, n_users)
    cold = rng.random(n_users) < 0.05
    activity[cold] = rng.integers(1, 4, cold.sum())
    activity *= target / activity.sum()
    counts = np.clip(np.round(activity), 1, served.sum() - 5).astype(int)
    rows = []
    for u in range(n_users):
        logits = 0.8 * np.log(pop) + 2.2 * affinity[u]
        logits[~served] -= 1.5  # explicit fictional entries are rarely rated (exercises the drop rule)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen = rng.choice(n_items, size=min(counts[u], n_items), replace=False, p=p)
        a_u = affinity[u, chosen]
        centred = (a_u - affinity[u].mean()) / (affinity[u].std() + 1e-9)
        r = bias[u] + 1.3 * centred + 0.55 * Q[chosen] + rng.normal(0, 0.9, len(chosen))
        r = np.clip(np.round(r), 1, 10).astype(int)
        for i, rv in zip(chosen, r, strict=True):
            rows.append((u + 1, int(i), int(rv)))
    ratings = pd.DataFrame(rows, columns=["offline_user_id", "item_idx", "rating"])
    log(f"[synthetic] users={n_users} ratings={len(ratings)} cold_users={int(cold.sum())}")

    # ------------------------------------------------------------------ collections (independent process)
    n_coll_users = int(cfg.get("n_collectors", 240))
    coll_users = rng.choice(n_users, size=min(n_coll_users, n_users), replace=False)
    collections = []
    cid = 0
    for u in coll_users:
        for _ in range(int(rng.integers(1, 4))):
            cid += 1
            theme = rng.choice(len(tag_vocab), size=2, replace=False)
            tv = np.zeros(len(tag_vocab))
            tv[theme] = 1.0
            ta = (Xn @ tv) + 0.35 * affinity[u] + 0.25 * np.log(pop) / np.log(pop).max()
            ta[~served] = -9
            p = np.exp(3.0 * (ta - ta.max()))
            p /= p.sum()
            size = int(rng.integers(4, 14))
            members_c = rng.choice(n_items, size=size, replace=False, p=p)
            name = f"{tag_vocab[theme[0]][1]} & {tag_vocab[theme[1]][1]} picks"
            collections.append({"collection_id": f"syn-col-{cid:04d}", "offline_user_id": int(u) + 1,
                                "name": name, "theme_tags": [list(tag_vocab[t]) for t in theme],
                                "items": [synthetic_key(seed, items[int(i)]["n"]) for i in members_c],
                                "label_source": "synthetic_generator"})

    # ------------------------------------------------------------------ catalogue records
    keys = [synthetic_key(seed, it["n"]) for it in items]
    char_rows, alias_rows, tag_rows, rel_rows, catalog_rows = [], [], [], [], []
    for i, it in enumerate(items):
        key = keys[i]
        cast = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(int(rng.integers(2, 5)))]
        synopsis = None
        if not it["explicit"] and rng.random() > 0.06:
            lead = cast[0]
            m_text = " and ".join(MOTIFS[m][0] for m in it["motifs"])
            gtxt = ", ".join(g.lower() for g in it["genres"])
            ttxt = (" Themes include " + ", ".join(t.lower() for t in it["themes"]) + ".") if it["themes"] else ""
            synopsis = (f"{lead} {m_text}. What begins as an ordinary story soon turns into a {gtxt} tale "
                        f"about the people around {lead.split()[0]}.{ttxt} (Synthetic record — fictional.)")
        per_ep = it["minutes"]
        eps = it["episodes"]
        total = round(per_ep * eps, 2) if (per_ep is not None and eps) else None
        missing = [f for f, v in (("score", it["score"]), ("episodes", eps), ("duration", per_ep),
                                  ("studios", it["studios"]), ("synopsis", synopsis)) if v in (None, [])]
        in_catalog = not it["explicit"]
        recommendable = in_catalog and it["score"] is not None and bool(it["genres"])
        planner_ok = recommendable and it["status"] == "Finished Airing" and eps is not None and per_ep is not None
        reasons = (["content_rule_explicit"] if it["explicit"] else []) + \
            (["not_recommendable_missing_score"] if in_catalog and it["score"] is None else [])
        catalog_rows.append({
            "item_key": key, "mal_id": None, "title": it["title"], "title_english": None,
            "title_other": None, "synopsis": synopsis, "score": it["score"],
            "scored_by": int(members[i] * 0.6) if it["score"] is not None else None, "rank": None,
            "popularity": None, "members": int(members[i]), "favorites": int(members[i] * 0.02),
            "type": it["type"], "episodes": eps, "duration_minutes": per_ep, "total_minutes": total,
            "duration_raw": (f"{int(per_ep)} min per ep" if (per_ep and it["type"] != "Movie") else
                             (f"{int(per_ep)} min" if per_ep else None)),
            "duration_basis": (None if per_ep is None else "total_runtime" if eps == 1 else "per_episode_average"),
            "status": it["status"], "content_rating": it["content_rating"], "source_material": it["source"],
            "aired_raw": str(it["year"]), "aired_from": str(it["year"]), "aired_to": None,
            "premiered_season": None, "premiered_year": None, "cover_url": None,
            "provenance": {"source": "synthetic", "seed": seed, "generator": PIPELINE_VERSION},
            "missing_fields": missing, "in_catalog": in_catalog, "is_recommendable": recommendable,
            "is_planner_eligible": planner_ok, "is_explicit": it["explicit"], "exclusion_reasons": reasons,
        })
        for ns, vals in (("genre", it["genres"]), ("theme", it["themes"]), ("demographic", it["demographic"]),
                         ("studio", it["studios"]), ("source", [it["source"]])):
            for v in vals:
                tag_rows.append({"item_key": key, "namespace": ns, "value": v, "source": "synthetic"})
        alias_rows.append({"item_key": key, "alias": it["title"], "kind": "title"})
        for c in cast:
            char_rows.append({"source_id": f"synthetic:{seed}:char:{key.split(':')[-1]}:{c}", "name": c,
                              "item_key": key, "role": "Main" if c == cast[0] else "Supporting",
                              "source_ref": f"synthetic:{seed}:cast:{key}"})
    for chain in franchises:
        for a, b in zip(chain, chain[1:], strict=False):
            ref = f"synthetic:{seed}:franchise:{keys[chain[0]]}"
            rel_rows.append({"from_item": keys[a], "to_item": keys[b], "to_entity_type": "anime",
                             "to_external_id": None, "to_name": items[b]["title"], "relation_type": "sequel",
                             "status": "confirmed", "source_ref": ref})
            rel_rows.append({"from_item": keys[b], "to_item": keys[a], "to_entity_type": "anime",
                             "to_external_id": None, "to_name": items[a]["title"], "relation_type": "prequel",
                             "status": "confirmed", "source_ref": ref})
    # an unresolved external edge (manga adaptation source) to exercise non-anime handling
    if franchises:
        rel_rows.append({"from_item": keys[franchises[0][0]], "to_item": None, "to_entity_type": "manga",
                         "to_external_id": f"synthetic-manga-{seed}-1", "to_name": items[franchises[0][0]]["title"],
                         "relation_type": "adaptation", "status": "confirmed",
                         "source_ref": f"synthetic:{seed}:adaptation"})

    # ------------------------------------------------------------------ scenes + fixture frames
    scene_rows, image_rows, scene_queries, image_queries = [], [], [], []
    scene_items = [i for i, it in enumerate(items) if not it["explicit"] and it["type"] == "TV"][: 12]
    sid = 0
    for i in scene_items:
        for _ in range(2):
            sid += 1
            m = MOTIFS[items[i]["motifs"][0]]
            ep = int(rng.integers(1, (items[i]["episodes"] or 12) + 1)) if rng.random() > 0.3 else None
            ts = int(rng.integers(60, 1400)) if (ep and rng.random() > 0.4) else None
            desc = (f"In a rain-soaked scene, the lead {m[0]} while a {rng.choice(['bell', 'kite', 'paper boat', 'clock tower', 'red umbrella'])} "
                    f"appears in the background. (Fictional scene.)")
            scene_rows.append({"scene_id": f"syn-scene-{sid:03d}", "item_key": keys[i], "description": desc,
                               "episode": ep, "timestamp_seconds": ts, "characters": [],
                               "source_ref": f"synthetic:{seed}:scene:{sid}", "source_url": None,
                               "spoiler_level": "minor", "verification_status": "verified"})
            scene_queries.append({"query_id": f"sq-{sid:03d}", "query": f"the scene where someone {m[1]} and there is "
                                  f"something in the background in the rain", "target_scene_id": f"syn-scene-{sid:03d}",
                                  "target_item_key": keys[i], "episode": ep, "label_source": "synthetic_generator"})
    img_dir = out / "raw" / "scene_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for k, row in enumerate(scene_rows[: int(cfg.get("n_scene_images", 12))]):
        png = _fixture_png(rng, k)
        path = img_dir / f"{row['scene_id']}.png"
        path.write_bytes(png)
        image_rows.append({"image_id": f"syn-img-{k + 1:03d}", "scene_id": row["scene_id"], "item_key": row["item_key"],
                           "path": str(path.relative_to(settings.data_root.parent)), "sha256": sha256_file(path),
                           "episode": row["episode"], "timestamp_seconds": row["timestamp_seconds"],
                           "split": "gallery", "provenance": {"source": "synthetic_fixture", "seed": seed}})
        if k < 6:
            qpath = out / "evaluation" / "image_queries" / f"q-{k + 1:03d}.png"
            qpath.parent.mkdir(parents=True, exist_ok=True)
            qpath.write_bytes(_fixture_png(rng, k, jitter=True))
            image_queries.append({"query_id": f"iq-{k + 1:03d}", "path": str(qpath.relative_to(settings.data_root.parent)),
                                  "sha256": sha256_file(qpath), "target_item_key": row["item_key"],
                                  "in_gallery": True, "label_source": "synthetic_fixture"})
    ood = out / "evaluation" / "image_queries" / "q-ood.png"
    ood.write_bytes(_fixture_png(rng, 999, jitter=True, ood=True))
    image_queries.append({"query_id": "iq-ood", "path": str(ood.relative_to(settings.data_root.parent)),
                          "sha256": sha256_file(ood), "target_item_key": None, "in_gallery": False,
                          "label_source": "synthetic_fixture"})

    # ------------------------------------------------------------------ evaluation files (held out of corpus)
    sim_labels = _similarity_labels(rng, items, keys, Xn, H, served)
    rag_queries = _rag_queries(rng, items, keys, rel_rows)
    system_tasks = _system_tasks(items, keys)
    ev = out / "evaluation"
    write_jsonl(ev / "similarity_labels.jsonl", sim_labels)
    write_jsonl(ev / "collection_events.jsonl", collections)
    write_jsonl(ev / "rag_queries.jsonl", rag_queries)
    write_jsonl(ev / "scene_queries.jsonl", scene_queries)
    write_jsonl(ev / "image_queries.jsonl", image_queries)
    write_jsonl(ev / "system_tasks.jsonl", system_tasks)

    raw = out / "raw"
    write_jsonl(raw / "anime.jsonl", catalog_rows)
    ratings.to_csv(raw / "ratings.csv", index=False)
    write_jsonl(raw / "tags.jsonl", tag_rows)
    write_jsonl(raw / "aliases.jsonl", alias_rows)
    write_jsonl(raw / "relations.jsonl", rel_rows)
    write_jsonl(raw / "characters.jsonl", char_rows)
    write_jsonl(raw / "scenes.jsonl", scene_rows)
    write_jsonl(raw / "scene_images.jsonl", image_rows)
    gen_manifest = {"generator": "pipelines.synthetic", "pipeline_version": PIPELINE_VERSION, "seed": seed,
                    "config": {k: cfg.get(k) for k in sorted(cfg)}, "created_at": utc_now_iso(),
                    "counts": {"anime": n_items, "users": n_users, "ratings": len(ratings),
                               "collections": len(collections), "scenes": len(scene_rows),
                               "scene_images": len(image_rows), "similarity_label_queries": len(sim_labels),
                               "rag_queries": len(rag_queries)},
                    "files": {str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*"))
                              if p.is_file() and p.name != "generator_manifest.json"}}
    write_json(out / "generator_manifest.json", gen_manifest)
    return gen_manifest


def _fixture_png(rng: np.random.Generator, k: int, jitter: bool = False, ood: bool = False) -> bytes:
    """Tiny deterministic colour-pattern frame (fixture only, not an anime frame)."""
    from PIL import Image

    base_rng = np.random.default_rng(1000 + k)
    colors = base_rng.integers(0, 255, size=(4, 3))
    if ood:
        colors = np.array([[250, 250, 250], [5, 5, 5], [250, 5, 250], [5, 250, 5]])
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for qi, (y, x) in enumerate([(0, 0), (0, 32), (32, 0), (32, 32)]):
        arr[y:y + 32, x:x + 32] = colors[qi]
    if jitter:
        noise = rng.integers(-18, 18, size=arr.shape)
        arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
        arr = arr[2:62, 3:63]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _similarity_labels(rng, items, keys, Xn, H, served) -> list[dict]:
    """Generator-truth similarity (tags + hidden latent + franchise). Pipeline validation only."""
    sims = Xn @ Xn.T + 0.5 * (H @ H.T) / H.shape[1]
    n = len(items)
    queries = [i for i in rng.permutation(n) if served[i] and items[i]["score"] is not None][:60]
    out = []
    for q in queries:
        s = sims[q].copy()
        s[q] = -np.inf
        s[~served] = -np.inf
        top = [int(j) for j in np.argsort(-s)[:4]]
        fr = [j for j in (items[q]["franchise"] or []) if j != q]
        pos = [{"item_key": keys[j], "grade": 3, "reason": "same_franchise"} for j in fr]
        for rank, j in enumerate(top):
            if j not in fr:
                pos.append({"item_key": keys[j], "grade": 2 if rank < 2 else 1, "reason": "latent_similarity"})
        out.append({"query_item": keys[q], "positives": pos, "label_source": "synthetic_generator",
                    "label_strength": "generator_truth"})
    return out


def _rag_queries(rng, items, keys, rel_rows) -> list[dict]:
    out = []
    served = [i for i, it in enumerate(items) if not it["explicit"]]
    # 1) paraphrased plot descriptions -> the item(s) with that motif
    motif_items: dict[int, list[int]] = {}
    for i in served:
        if items[i]["motifs"] and items[i]["score"] is not None:
            for m in items[i]["motifs"]:
                motif_items.setdefault(m, []).append(i)
    for m, idxs in sorted(motif_items.items()):
        out.append({"query_id": f"rq-plot-{m:02d}", "query": f"an anime where the main character {MOTIFS[m][1]}",
                    "intent": "preference_recommendation", "relevant_item_keys": [keys[i] for i in idxs],
                    "required_facts": [], "hard_constraints": {}, "answerable": True,
                    "label_source": "synthetic_generator"})
    # 2) title lookup with a typo
    for i in [j for j in rng.permutation(served)[:10]]:
        t = items[i]["title"]
        pos = int(rng.integers(1, len(t) - 1))
        typo = t[:pos] + t[pos + 1:]
        out.append({"query_id": f"rq-title-{i:03d}", "query": f"tell me about {typo}", "intent": "anime_fact",
                    "relevant_item_keys": [keys[i]], "required_facts": ["episodes", "score"],
                    "hard_constraints": {}, "answerable": True, "label_source": "synthetic_generator"})
    # 3) constraint queries
    for g in ["Comedy", "Mystery", "Sci-Fi", "Romance", "Action"]:
        rel = [keys[i] for i in served if g in items[i]["genres"] and items[i]["episodes"]
               and items[i]["episodes"] <= 13 and items[i]["score"] is not None]
        out.append({"query_id": f"rq-cons-{g}", "query": f"a {g.lower()} anime with at most 13 episodes",
                    "intent": "preference_recommendation", "relevant_item_keys": rel, "required_facts": [],
                    "hard_constraints": {"include_genres": [g], "max_episodes": 13}, "answerable": bool(rel),
                    "label_source": "synthetic_generator"})
    # 4) relation queries
    for r in [r for r in rel_rows if r["relation_type"] == "sequel"][:6]:
        src = keys.index(r["from_item"])
        out.append({"query_id": f"rq-rel-{src:03d}", "query": f"what is the sequel to {items[src]['title']}?",
                    "intent": "relation_query", "relevant_item_keys": [r["to_item"]], "required_facts": ["sequel"],
                    "hard_constraints": {}, "answerable": True, "label_source": "synthetic_generator"})
    # 5) unanswerable / contradictory
    no_scene = [i for i in served if items[i]["type"] == "Movie"][:3]
    for i in no_scene:
        out.append({"query_id": f"rq-unans-{i:03d}", "query": f"which episode of {items[i]['title']} has the "
                    f"rooftop confession?", "intent": "scene_lookup", "relevant_item_keys": [],
                    "required_facts": ["episode"], "hard_constraints": {}, "answerable": False,
                    "label_source": "synthetic_generator"})
    out.append({"query_id": "rq-contra-1", "query": "a long series with at least 50 episodes but at most 3 episodes",
                "intent": "preference_recommendation", "relevant_item_keys": [], "required_facts": [],
                "hard_constraints": {"min_episodes": 50, "max_episodes": 3}, "answerable": False,
                "label_source": "synthetic_generator"})
    return out


def _system_tasks(items, keys) -> list[dict]:
    return [
        {"task_id": "st-chat-constraint", "surface": "chat", "steps": ["send: a comedy anime with at most 12 episodes"],
         "success": {"all_items": {"max_episodes": 12, "include_genres": ["Comedy"]}, "min_items": 1}},
        {"task_id": "st-chat-refine", "surface": "chat",
         "steps": ["send: something like " + items[0]["title"], "send: at most 13 episodes"],
         "success": {"all_items": {"max_episodes": 13}, "exclude_items": [keys[0]]}},
        {"task_id": "st-plan-budget", "surface": "plan", "steps": ["plan: budget_minutes=900 max_titles=3"],
         "success": {"total_minutes_lte": 900, "max_titles": 3}},
    ]


def preprocess_synthetic(settings: Settings, log=print) -> dict[str, Any]:
    """Convert generator output into the shared processed layout."""
    src = settings.synthetic_dir
    gen_manifest_path = src / "generator_manifest.json"
    if not gen_manifest_path.exists():
        raise FileNotFoundError("synthetic data not generated yet (run: anilogic generate-synthetic)")
    gen_manifest = json.loads(gen_manifest_path.read_text(encoding="utf-8"))
    # hash the generated data files (not the manifest, which carries a timestamp)
    source_hashes = {"generator_files": sha256_json(gen_manifest["files"])}
    version = dataset_version("synthetic", source_hashes, settings.fingerprint("content_rules"))
    root = settings.processed_dir / version
    writer = ProcessedWriter(root, int(settings.pipeline.get("rating_partitions", 4)))

    def jl(name: str) -> list[dict]:
        p = src / "raw" / name
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    catalog = pd.DataFrame(jl("anime.jsonl"))
    writer.write_catalog(catalog)
    tags = pd.DataFrame(jl("tags.jsonl"))
    writer.write_table("item_tags", tags)
    writer.write_table("aliases", pd.DataFrame(jl("aliases.jsonl")))
    writer.write_table("relations", pd.DataFrame(jl("relations.jsonl")))
    writer.write_table("characters", pd.DataFrame(jl("characters.jsonl")))
    writer.write_table("scenes", pd.DataFrame(jl("scenes.jsonl")))
    writer.write_table("scene_images", pd.DataFrame(jl("scene_images.jsonl")))
    ratings = pd.read_csv(src / "raw" / "ratings.csv")
    served = catalog["in_catalog"].to_numpy()
    valid = ratings["rating"].between(1, 10) & ratings["item_idx"].between(0, len(catalog) - 1)
    keep = valid & served[ratings["item_idx"].clip(0, len(catalog) - 1)]
    kept = ratings[keep]
    writer.append_ratings(kept.offline_user_id.to_numpy(), kept.item_idx.to_numpy(), kept.rating.to_numpy())
    dedupe = writer.close_ratings()
    report = {"dataset_version": version, "catalog": {
        "parsed": len(catalog), "in_catalog": int(catalog.in_catalog.sum()),
        "recommendable": int(catalog.is_recommendable.sum()),
        "planner_eligible": int(catalog.is_planner_eligible.sum()),
        "explicit_excluded": int(catalog.is_explicit.sum())},
        "ratings": {"rows_read": len(ratings), "invalid": int((~valid).sum()),
                    "dropped_content_rule_item": int((valid & ~keep).sum()), **dedupe,
                    "users": int(kept.offline_user_id.nunique())},
        "tags_per_namespace": Counter(tags.namespace).most_common()}
    writer.write_report("preprocess_report", report)
    manifest = writer.finalize({
        "dataset_mode": "synthetic", "dataset_version": version, "pipeline_version": PIPELINE_VERSION,
        "source_hashes": source_hashes, "created_at": utc_now_iso(), "seed": settings.synthetic.get("seed", 42),
        "counts": {"catalog": len(catalog), "ratings": dedupe["rows"], "users": int(kept.offline_user_id.nunique())},
        "dependencies": dependency_versions()})
    # copy held-out evaluation files next to the processed version (read-only use by evaluation/)
    write_json(settings.processed_dir / "latest.json", {"dataset_version": version, "updated_at": utc_now_iso()})
    log(f"[preprocess-synthetic] {version}: {len(catalog)} titles, {dedupe['rows']} ratings")
    return manifest
