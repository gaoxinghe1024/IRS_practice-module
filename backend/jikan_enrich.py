# -*- coding: utf-8 -*-
"""
Jikan API 数据补全脚本 —— 数据 pipeline 第一步（对应项目缺口分析的最高优先级事项）。

为 anime 表补全：
    1. mal_id      —— 种子内置官方 ID 映射（或标题搜索兜底）
    2. image_url   —— 官方封面图（前端直接展示真图）
    3. themes      —— 官方主题标签（英文），经内置映射翻成中文后合并进现有标签
    4. demographics—— 官方受众分级（Shounen / Seinen / ...）
    5. characters  —— 主要角色名（现有角色为空时回填）

用法：
    python jikan_enrich.py            # 全量补全
    python jikan_enrich.py --limit 3  # 只补 3 部（测试）
    python jikan_enrich.py --force    # 已补全的也重刷

特性：
    - 限速：Jikan 公共 API 约 3 req/s，脚本统一 0.45s 间隔；
    - 429/5xx 自动退避重试（Jikan 上游 MyAnimeList 偶发抖动）；
    - 按 ID 直查后做标题校验，防错配；
    - 幂等：已补全（mal_id>0）的条目默认跳过。
"""
import json
import sys
import time
import urllib.parse
import urllib.request

from database import get_conn

BASE = "https://api.jikan.moe/v4"
DELAY = 0.45

# Jikan 英文标签 -> 中文映射（覆盖常见 Themes / Demographics / Genres）
TAG_ZH = {
    "Psychological": "心理", "Adult Cast": "成人主角", "Shounen": "少年向", "Shoujo": "少女向",
    "Seinen": "青年向", "Josei": "女性向", "Kids": "儿童向", "Isekai": "异世界",
    "Time Travel": "时间旅行", "Time Loop": "时间循环", "School": "校园", "Military": "军事",
    "Super Power": "超能力", "Violence": "暴力", "Survival": "生存", "Gore": "猎奇",
    "Historical": "历史", "Mecha": "机甲", "Music": "音乐", "Sports": "运动",
    "Love Polygon": "三角恋", "Reverse Harem": "逆后宫", "Harem": "后宫", "Idols (Female)": "偶像(女)",
    "Idols (Male)": "偶像(男)", "Workplace": "职场", "Otaku Culture": "御宅文化",
    "Organized Crime": "黑帮", "Detective": "侦探", "Vampire": "吸血鬼", "Zombie": "丧尸",
    "Mythology": "神话", "Samurai": "武士", "Reincarnation": "转生", "Found Family": "羁绊家庭",
    "Performing Arts": "表演艺术", "Delinquents": "不良少年", "Crossdressing": "伪娘",
    "Educational": "教育", "Racing": "竞速", "Strategy Game": "策略博弈", "High Stakes Game": "死亡游戏",
    "Team Sports": "团队运动", "Combat Sports": "格斗竞技", "Anthropomorphic": "拟人化",
    "CGDCT": "美少女日常", "Gag Humor": "搞笑", "Showbiz": "演艺圈", "Space": "宇宙",
    "Superhero": "英雄", "Visual Arts": "视觉艺术", "Medical": "医疗", "Pets": "宠物",
    "Romantic Subtext": "暧昧", "Cyberpunk": "赛博朋克", "Post-Apocalyptic": "末世",
    "Coming of Age": "青春成长", "Dark Fantasy": "黑暗奇幻", "Iyashikei": "治愈",
    "Action": "动作", "Adventure": "冒险", "Comedy": "喜剧", "Drama": "剧情", "Fantasy": "奇幻",
    "Horror": "恐怖", "Mystery": "悬疑", "Romance": "爱情", "Sci-Fi": "科幻", "Slice of Life": "日常",
    "Supernatural": "超自然", "Suspense": "悬疑", "Award Winning": "获奖作品",
    "Avant Garde": "先锋", "Boys Love": "BL", "Girls Love": "GL",
    "Childcare": "育儿", "Parody": "戏仿恶搞",
}

_seen_tags: set[str] = set()  # 收集未翻译标签，便于人工补映射

# 种子动漫的 MyAnimeList 官方 ID（按 ID 直查，绕开搜索接口；运行时校验标题防错配）
SEED_MAL_IDS = {
    "Death Note": 1535, "Shingeki no Kyojin": 16498,
    "Fullmetal Alchemist: Brotherhood": 5114, "Steins;Gate": 9253,
    "Code Geass: Hangyaku no Lelouch": 1575, "Monster": 19,
    "Hunter x Hunter (2011)": 11061, "Kimetsu no Yaiba": 38000,
    "Jujutsu Kaisen": 40748, "Vinland Saga": 37521,
    "Cowboy Bebop": 1, "Neon Genesis Evangelion": 30,
    "Mushishi": 457, "Natsume Yuujinchou": 4081,
    "One Punch Man": 30276, "Spy x Family": 50265,
    "Mob Psycho 100": 32182, "Higurashi no Naku Koro ni": 934,
    "Gintama": 918, "Kimi no Na wa": 32281,
    "Re:Zero kara Hajimeru Isekai Seikatsu": 31240,
    "Shigatsu wa Kimi no Uso": 23273, "Hataraku Saibou": 35860,
    # "L: Change the World" 为真人电影，不在 MAL 动漫库中，预期跳过
}


def _zh(tag: str) -> str:
    _seen_tags.add(tag)
    return TAG_ZH.get(tag, tag)


def _title_ok(expected: str, hit: dict) -> bool:
    names = " | ".join(filter(None, [hit.get("title"), hit.get("title_english"),
                                     hit.get("title_japanese")])).lower()
    e = expected.lower()
    return e in names or (names.split(" | ")[0] and names.split(" | ")[0] in e)


def _curl_get(path: str) -> dict | None:
    """备用通道：curl 子进程（部分环境下 urllib 与 curl 的网络出口表现不同）。"""
    import subprocess
    url = BASE + path
    try:
        p = subprocess.run(["curl", "-s", "--max-time", "12", url],
                           capture_output=True, text=True, timeout=20)
        if p.returncode == 0 and p.stdout.strip():
            return json.loads(p.stdout)
    except Exception as e:
        print(f"    [warn] curl {type(e).__name__}")
    return None


def _get(path: str, retries: int = 3) -> dict | None:
    url = BASE + path
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AnimeFinderMVP/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and i < retries:
                time.sleep(1.5 + i)   # 上游抖动退避
                continue
            print(f"    [warn] HTTP {e.code}: {url}")
            break
        except Exception as e:
            print(f"    [warn] {type(e).__name__}: {url}")
            break
    return _curl_get(path)  # urllib 失败时 curl 兜底


def _info_from_detail(hit: dict, expected: str) -> dict | None:
    if not _title_ok(expected, hit):
        print(f"    [warn] 标题不匹配: 期望 {expected!r}, 返回 {hit.get('title')!r}，跳过")
        return None
    return {
        "mal_id": hit["mal_id"],
        "image_url": (hit.get("images", {}).get("jpg", {}) or {}).get("image_url", ""),
        "themes_zh": [_zh(t["name"]) for t in hit.get("themes", [])],
        "demographics": _zh(hit["demographics"][0]["name"]) if hit.get("demographics") else "",
        "characters_zh": [],
    }


def _attach_characters(info: dict) -> dict:
    chars = _get(f"/anime/{info['mal_id']}/characters")
    if chars and chars.get("data"):
        mains = [c for c in chars["data"] if c.get("role") == "Main"] or chars["data"]
        info["characters_zh"] = [c["character"]["name"] for c in mains[:4]]
    return info


def fetch_anime_info(title: str) -> dict | None:
    """优先按内置 ID 直查详情（稳定），否则标题搜索兜底。"""
    if title in SEED_MAL_IDS:
        data = _get(f"/anime/{SEED_MAL_IDS[title]}")
        if data and data.get("data"):
            info = _info_from_detail(data["data"], title)
            if info:
                return _attach_characters(info)
    q = urllib.parse.quote(title)
    data = _get(f"/anime?q={q}&limit=3")
    if not data or not data.get("data"):
        return None
    hit = next((d for d in data["data"] if _title_ok(title, d)), None)
    if not hit:
        return None
    info = _info_from_detail(hit, title)
    return _attach_characters(info) if info else None


def run(limit: int | None = None, force: bool = False):
    conn = get_conn()
    if force:
        rows = conn.execute("SELECT id,title FROM anime").fetchall()
    else:
        rows = conn.execute("SELECT id,title FROM anime WHERE mal_id=0 OR mal_id IS NULL").fetchall()
    if limit:
        rows = rows[:limit]

    ok = fail = 0
    for r in rows:
        aid, title = r["id"], r["title"]
        print(f"[{aid}] {title} …")
        info = fetch_anime_info(title)
        if not info:
            fail += 1
            continue

        cur = conn.execute("SELECT themes, demographics, characters FROM anime WHERE id=?", (aid,)).fetchone()
        old_themes = [t.strip() for t in cur["themes"].split(",") if t.strip()]
        merged = list(dict.fromkeys(old_themes + info["themes_zh"]))
        demo = cur["demographics"] or info["demographics"]
        chars = [c.strip() for c in cur["characters"].split(",") if c.strip()] or info["characters_zh"]

        conn.execute(
            """UPDATE anime SET mal_id=?, image_url=?, themes=?, demographics=?, characters=? WHERE id=?""",
            (info["mal_id"], info["image_url"], ",".join(merged), demo, ",".join(chars), aid),
        )
        conn.commit()
        ok += 1
        print(f"    ok  mal_id={info['mal_id']} themes={info['themes_zh']} demo={demo or '-'}")
        time.sleep(DELAY)

    conn.close()
    print(f"\n完成：成功 {ok}，失败 {fail}")
    untranslated = [t for t in _seen_tags if t not in TAG_ZH]
    if untranslated:
        print("未翻译标签（可补充进 TAG_ZH）:", ", ".join(sorted(untranslated)))


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    force = "--force" in args
    run(limit=limit, force=force)
