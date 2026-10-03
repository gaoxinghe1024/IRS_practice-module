# -*- coding: utf-8 -*-
"""score.py — rule scorers and title extraction for the Page-4 RAG evaluation.

This is the FINAL (consolidated) version of every rule scorer used in the
evaluation, including all later corrections:

* title extraction        - deterministic, alias-table based, longest-match-first;
                            short (<=2 char) aliases only hit inside quotes;
                            Latin-script aliases also need a word boundary in
                            the original text (ASCII letters/digits do not count
                            as boundaries).
* episode_ok / top1/top3  - scene metrics (KB-consistent episode accuracy);
                            disputed KB entries S08/S12 each have TWO phrasings
                            (S08e/S08h/S12e/S12h) - see DISPUTED_SCENE_QIDS.
* all_satisfy             - constraint metric with per-work satisfaction.
* fact_ok (final)         - EN+ZH keyword lexicon for Movie/TV, Chinese-numeral
                            fallback for episodes, negation guard for aired/movie.
* rule_frank (final)      - FRANK_V2 regex (v1 kept for reference/sensitivity);
                            T10 (offtopic) is reported separately, never pooled.
* echo / in-KB variants   - reply-level in-KB citation caliber: query-echo titles
                            and in-KB spelling variants are removed first.

CLI (all paths relative to the package root, pathlib, UTF-8):
    python harness/score.py --results-dir results --out results/scored_rebuilt.csv
Default mode is REPLAY: it re-scores the saved replies in results/raw.jsonl.gz
and never calls any model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sqlite3
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- text utils
FW = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
FW_STR = str.maketrans("０１２３４５６７８９", "0123456789")
PUNCT = re.compile(r"[\s。．·：:；;，,、！!？?（）()\[\]【】\-—_~/*&\"'“”‘’.]+")

MOVIE_RE = re.compile(r"电影|剧场版|劇場版|映画|movie|film", re.I)
TV_RE = re.compile(r"\bTV\b|电视|テレビ", re.I)
NEG_RE = re.compile(r"(不是|并非|而不是|not a|not an)[^。]{0,8}$", re.I)
CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
          "八": 8, "九": 9}

FRANK_V1 = re.compile(r"查不到|找不到|不在库内|没有收录|未收录|不了解|无法|没有这部|库里没有|不知道")
FRANK_V2 = re.compile(
    r"查不到|找不到|不了解|不知道(?![^。]{0,12}不影响)|没有这部|"
    r"不在[^。]{0,8}(知识库|资料库|库内|库里|库中)|"
    r"(知识库|资料库|库里|库中|手头的知识库)[^。]{0,10}没有|"
    r"没有收录|未收录|并未收录|"
    r"无法|没法|没办法|帮不上忙|答不上来|"
    r"(不能|不想|不太想|不会|不便|无法)[^。]{0,6}(凭空|随便|随意|凭印象)?(编造|瞎编|胡编|乱编|猜测|编一个|瞎猜|乱编)|"
    r"没有?(查到|找到|检索到|搜到)|检索不到|"
    r"没有[^。]{0,8}(相关的?|官方的?)?(资料|信息|记录|数据)|没有[^。]{0,12}相关的?内容|没有关于|"
    r"没有把握|不敢(确定|乱说|随便|瞎编|肯定|硬编)|"
    r"超出[^。]{0,4}(范围|能力)|与动漫无关|非动漫",
    re.I)

DISPUTED_SCENE_QIDS = {"S08e", "S08h", "S12e", "S12h"}  # KB-vs-real discrepancies, both phrasings
OFFTOPIC_TRAP_QID = "T10"  # offtopic trap: refuse-and-say-out-of-scope, reported separately


def norm_text(s: str) -> str:
    return PUNCT.sub("", str(s).translate(FW_STR).lower())


def num_present(text: str, n: int) -> bool:
    return re.search(rf"(?<!\d){re.escape(str(n))}(?!\d)", text) is not None


def cn_to_int(s: str):
    if "十" in s:
        a, _, b = s.partition("十")
        return CN_NUM.get(a, 1) * 10 + (CN_NUM.get(b, 0) if b else 0)
    if "百" in s:
        a, _, rest = s.partition("百")
        v = CN_NUM.get(a, 0) * 100
        if rest:
            v += cn_to_int(rest)
        return v
    return CN_NUM.get(s)


# ------------------------------------------------------------- title extraction
_NORM_STRIP = re.compile(r"[\s。．・·：:；;，,、！!？?\"'“”‘’（）()\[\]【】\-—_~×x×./*&]+", re.I)
_QUOTE_SPAN = re.compile(r"([《「『\"“])([^《》「」『』\"”]{1,40}?)([》」』\"”])")
_BOOK_SPAN = re.compile(r"《([^《》]{1,40}?)》")

# Verbatim norm() of the evaluation's aliases module: lower-case, strip season /
# year-edition / part / movie suffixes, then strip punctuation (incl. "x" and "×",
# so "SPY×FAMILY" == "spyfamily"). Used ONLY inside mentioned_animes, where the
# original extractor applied it; norm_text (no suffix stripping) stays elsewhere.
_SUFFIX_SEASON = re.compile(r"第[一二三四五六七八九十\d]+季(度)?$")
_SUFFIX_EDITION = re.compile(r"(\(\d{4}\s*(版)?\)|（\d{4}\s*(版)?）|\d{4}版)$")
_SUFFIX_MISC = re.compile(r"(2011版|第二部|剧场版)$")


def norm_title(s: str) -> str:
    t = str(s).lower()
    t = _SUFFIX_SEASON.sub("", t)
    t = _SUFFIX_EDITION.sub("", t)
    t = _SUFFIX_MISC.sub("", t)
    t = _NORM_STRIP.sub("", t)
    return t


def load_aliases(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "aliases" in data and isinstance(data["aliases"], dict):
        data = data["aliases"]
    return data


def _index_map(original: str) -> tuple[str, list[int]]:
    nt, idx = [], []
    for i, ch in enumerate(original):
        if _NORM_STRIP.fullmatch(ch):
            continue
        nt.append(ch.lower())
        idx.append(i)
    return "".join(nt), idx


def _ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def mentioned_animes(answer: str, aliases: dict) -> dict:
    """Deterministic title extraction (longest-match-first, non-overlapping).

    Short aliases (normalized length < 3) only match inside quotes/brackets.
    Latin-script aliases (pure-ASCII normalized form) also need a word boundary
    in the original text: the characters just before and after the match must
    not be ASCII letters or digits (CJK characters, punctuation, whitespace and
    the string ends all count as boundaries), so "NGE" does not hit "range(".
    《…》 spans that match no KB alias are returned as out_of_kb.
    """
    hits: list[tuple[str, int]] = []
    taken: list[tuple[int, int]] = []
    for m in _QUOTE_SPAN.finditer(answer):
        inner_norm = norm_title(m.group(2))
        if not inner_norm:
            continue
        for title, entries in aliases.items():
            for e in entries:
                if e["norm"] == inner_norm:
                    hits.append((title, m.start(2)))
                    taken.append((m.start(), m.end()))
                    break
            else:
                continue
            break

    def overlapped(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in taken)

    nt, idx = _index_map(answer)
    flat = [(e["norm"], title) for title, entries in aliases.items() for e in entries]
    flat.sort(key=lambda t: -len(t[0]))
    occupied = [False] * len(nt)

    def range_taken(a: int, b: int) -> bool:
        if any(occupied[a:b]):
            return True
        if idx:
            oa, ob = idx[a], idx[min(b, len(idx) - 1)] + 1
            if overlapped(oa, ob):
                return True
        return False

    def word_boundary(p: int, end: int) -> bool:
        before = answer[idx[p] - 1] if idx[p] > 0 else ""
        after_pos = idx[end - 1] + 1
        after = answer[after_pos] if after_pos < len(answer) else ""
        return not (before and _ascii_alnum(before)) \
            and not (after and _ascii_alnum(after))

    for na, title in flat:
        if len(na) < 3:
            continue
        latin = na.isascii()
        start = 0
        while True:
            p = nt.find(na, start)
            if p < 0:
                break
            end = p + len(na)
            if latin and not word_boundary(p, end):
                start = p + 1
            elif not range_taken(p, end):
                for k in range(p, end):
                    occupied[k] = True
                hits.append((title, idx[p]))
                start = end
            else:
                start = p + 1

    hits.sort(key=lambda t: t[1])
    kb_hits, seen = [], set()
    for title, _pos in hits:
        if title not in seen:
            seen.add(title)
            kb_hits.append(title)

    out_of_kb = []
    for m in _BOOK_SPAN.finditer(answer):
        inner = norm_title(m.group(1))
        if not inner:
            continue
        if not any(e["norm"] == inner for entries in aliases.values() for e in entries):
            if m.group(1) not in out_of_kb:
                out_of_kb.append(m.group(1))
    return {"kb_hits": kb_hits, "out_of_kb": out_of_kb}


def query_titles(qid: str, q: dict, aliases: dict) -> set:
    """Titles named by the query: 《…》/quotes + KB aliases + latin runs + manual map.

    Manual map covers trap questions T05-T09 whose out-of-KB titles appear bare
    (human-confirmed during the evaluation).
    """
    MANUAL = {"T05": ["One Piece", "海贼王"], "T06": ["火影忍者"], "T07": ["灌篮高手"],
              "T08": ["葬送的芙莉莲"], "T09": ["Detective Conan", "名侦探柯南"]}
    query = q["query"]
    tn = norm_text(query)
    titles = set(MANUAL.get(qid, []))
    for m in re.finditer(r"《([^《》]{1,40}?)》", query):
        titles.add(m.group(1))
    for m in re.finditer(r"[「\"“]([^「」\"”]{1,40}?)[」\"”]", query):
        titles.add(m.group(1))
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9:;·'&! ]{1,40}", query):
        titles.add(m.group(0).strip())
    for title, entries in aliases.items():
        for e in entries:
            na = e["norm"]
            if len(na) >= 2 and na in tn:
                titles.add(title)
                break
            if len(na) < 2 and re.search(rf"[《「『\"“]{re.escape(e['alias'])}[》」』\"”]", query):
                titles.add(title)
                break
    return titles


def is_echo(name: str, qtitles: set) -> bool:
    """True when a title merely repeats (a variant of) the query's own title."""
    n1 = norm_text(name)
    if len(n1) < 2:
        return False
    for t in qtitles:
        n2 = norm_text(t)
        if not n2:
            continue
        if n1 == n2 or n1 in n2 or n2 in n1:
            return True
    return False


def is_kb_variant(name: str, aliases: dict) -> bool:
    """True when an out-of-KB extracted name is a spelling variant of a KB title."""
    n1 = norm_text(name)
    if len(n1) < 2:
        return False
    for entries in aliases.values():
        for e in entries:
            n2 = e["norm"]
            if len(n2) < 2:
                continue
            if n1 == n2 or n1 in n2 or n2 in n1:
                return True
    return False


# ------------------------------------------------------------------- scorers
def fact_ok(field: str, val: str, reply: str) -> tuple[bool, str]:
    """Final fact scorer. Returns (ok, note)."""
    txt = str(reply).translate(FW_STR)
    if field == "episodes":
        if num_present(txt, int(val)):
            return True, "arabic digit"
        for m in re.finditer(r"([一二两三四五六七八九十百]{1,7})\s*[集话話]", txt):
            v = cn_to_int(m.group(1))
            if v == int(val):
                return True, f"chinese numeral {m.group(1)}"
        return False, "no match"
    if field == "aired":
        if str(val) in txt:
            i = txt.find(str(val))
            if NEG_RE.search(txt[max(0, i - 12):i]):
                return False, "negation context"
            return True, "literal year"
        return False, "no match"
    if field == "studios":
        return (norm_text(val) in norm_text(reply), "normalized containment")
    # an_type
    is_movie_ref = val == "Movie"
    m_hit = MOVIE_RE.search(reply)
    t_hit = TV_RE.search(reply)
    if is_movie_ref:
        if m_hit:
            if NEG_RE.search(reply[max(0, m_hit.start() - 12):m_hit.start()]):
                return False, "negation context"
            return True, "movie synonym (en+zh)"
        return False, "no movie keyword"
    if t_hit:
        return True, "tv keyword"
    return False, "no tv keyword"


def rule_frank(reply: str, version: int = 2) -> bool:
    pat = FRANK_V2 if version == 2 else FRANK_V1
    return bool(pat.search(reply or ""))


def satisfy_constraints(r: dict, cons: dict) -> bool:
    if cons.get("max_episodes") is not None:
        if r["episodes"] is None or r["episodes"] > cons["max_episodes"]:
            return False
    if cons.get("min_score") is not None:
        if r["score"] is None or r["score"] < cons["min_score"]:
            return False
    if cons.get("type"):
        if (r["an_type"] or "") != cons["type"]:
            return False
    if cons.get("tags_any"):
        pool = set((r["genres"] or "").split(",")) | set((r["themes"] or "").split(","))
        pool.discard("")
        if not (set(cons["tags_any"]) & pool):
            return False
    return True


# ---------------------------------------------------------------------- CLI
def load_jsonl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_anime(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = {r["title"]: dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()}
    conn.close()
    return rows


def score_records(raw: list[dict], testset: dict, aliases: dict, anime: dict,
                  retrieval: dict) -> list[dict]:
    """Score every saved reply. Pure function - no I/O, used by tests."""
    scored = []
    for r in raw:
        qid, group = r["qid"], r["group"]
        q = testset[qid]
        typ, exp = q["type"], q.get("expect", {})
        reply = r.get("reply") or ""
        m = mentioned_animes(reply, aliases)
        kb_hits = list(m["kb_hits"])
        if group == "T" and r.get("scene_focus_title") \
                and r["scene_focus_title"] not in kb_hits:
            kb_hits.insert(0, r["scene_focus_title"])
        out_kb = m["out_of_kb"]
        row = {"qid": qid, "type": typ, "model_label": r.get("model_label"),
               "group": group, "status": r.get("status"),
               "reply_len": len(reply), "n_kb_hits": len(kb_hits),
               "n_out_kb": len(out_kb), "kb_hits": ";".join(kb_hits),
               "out_kb": ";".join(out_kb),
               "input_tokens": r.get("input_tokens"),
               "output_tokens": r.get("output_tokens"),
               "duration_ms": r.get("duration_ms")}
        ok = r.get("status") == "ok" and bool(reply.strip())
        if typ == "scene" and ok:
            t = exp["title"]
            row["top1"] = bool(kb_hits and kb_hits[0] == t)
            row["top3"] = bool(t in kb_hits[:3])
            if anime.get(t, {}).get("an_type") != "Movie":
                row["episode_ok"] = num_present(reply.translate(FW_STR), exp["episode"])
        elif typ == "constraint" and ok:
            cons = exp.get("constraints", {})
            works = [w for w in kb_hits if w in anime]
            sat = [satisfy_constraints(anime[w], cons) for w in works]
            n_out = len(out_kb)
            row["n_rec"] = len(works) + n_out
            row["n_rec_in_kb"] = len(works)
            row["n_rec_satisfy"] = sum(sat)
            row["all_satisfy"] = bool(works) and all(sat) and n_out == 0
        elif typ == "mood" and ok:
            cons = {"tags_any": exp.get("tags_any"),
                    "max_episodes": exp.get("max_episodes")}
            row["mood_ok"] = any(w in anime and satisfy_constraints(anime[w], cons)
                                 for w in kb_hits)
        elif typ == "fact" and ok:
            f_ok, note = fact_ok(exp["field"], exp["value"], reply)
            row["fact_ok"] = f_ok
            row["fact_note"] = note
        elif typ == "trap" and ok:
            row["out_of_kb_answer"] = bool(out_kb)
            row["rule_frank"] = rule_frank(reply, 2)
            row["rule_frank_v1"] = rule_frank(reply, 1)
        # reply-level in-KB citation caliber (echo + in-KB variants removed first)
        # Final caliber: the reply counts as in-KB-compliant when, after removing
        # query echoes and in-KB spelling variants, no out-of-KB title remains AND
        # the reply recognizes at least one work title (KB hit or out-of-KB) —
        # replies recognizing zero titles are not "in-KB citations".
        if group in ("R", "N", "T") and ok:
            qt = query_titles(qid, q, aliases)
            kept = [s for s in out_kb
                    if not is_echo(s, qt) and not is_kb_variant(s, aliases)]
            row["in_kb_reply_ok"] = (len(kept) == 0
                                     and (len(kb_hits) + len(out_kb)) > 0)
        scored.append(row)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay rule scoring over saved replies "
                                             "(no model calls).")
    ap.add_argument("--results-dir", default="results",
                    help="package results directory (default: results)")
    ap.add_argument("--data-dir", default="data", help="package data directory")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <results-dir>/scored_rebuilt.csv)")
    args = ap.parse_args()

    pkg = PKG
    results = (pkg / args.results_dir) if not Path(args.results_dir).is_absolute() \
        else Path(args.results_dir)
    data = (pkg / args.data_dir) if not Path(args.data_dir).is_absolute() \
        else Path(args.data_dir)

    testset = {q["id"]: q for q in load_jsonl(data / "testset_v1.jsonl")}
    aliases = load_aliases(data / "aliases.json")
    anime = load_anime(data / "kb" / "anime.db")
    retrieval = {x["id"]: x["retrieval"]
                 for x in load_jsonl(results / "retrieval.jsonl")}
    raw = load_jsonl(results / "raw.jsonl.gz")
    scored = score_records(raw, testset, aliases, anime, retrieval)

    out = Path(args.out) if args.out else results / "scored_rebuilt.csv"
    cols: list[str] = []
    for row in scored:
        for k in row:
            if k not in cols:
                cols.append(k)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(scored)
    print(f"scored {len(scored)} replies -> {out}")
    print("SCORE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
