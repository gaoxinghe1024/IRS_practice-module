# -*- coding: utf-8 -*-
"""
Kaggle『Anime Dataset 2023』(dbdmobile/myanimelist-dataset) 接入脚本。
对应 database.load_mal_dataset 接口的正式实现（此前为 NotImplementedError）。

数据文件（Kaggle 下载）：
    anime-dataset-2023.csv   作品表（约 2.8 万行，含 Synopsis）
    users-score-2023.csv     评分表（约 2400 万行，1.2GB）
    users-details-2023.csv   用户表（可选，本脚本暂只做用户名登记）

用法（在 backend/ 目录下）：
    python load_mal.py --anime anime-dataset-2023.csv --scores users-score-2023.csv
    python load_mal.py --anime ... --scores ... --limit-ratings 3000000 --max-anime 12000

要点（与项目文档的数据分析对应）：
    1) 分块读取：评分表 2400 万行用 pandas chunksize=200000 流式处理，内存受限可跑；
    2) 过滤：剔除无评分(Unknown)作品、Hent*ai 等成人内容类型（可用作品约 1.2 万部）；
    3) 列名容错：Kaggle 表头可能有变，内置候选列名映射；
    4) Demographic（Shounen/Seinen 等）从 Genres 列拆出，英文主题标签经映射转中文；
    5) 载入后建议接着跑 jikan_enrich.py 补 themes/characters 的缺口。
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database  # noqa: E402  (DB_PATH 支持 ANIMEDB_PATH 环境变量覆盖)

try:
    import pandas as pd
except ImportError:
    sys.exit("需要 pandas：pip install pandas")

CHUNK = 200_000

# 各字段的候选列名（容错 Kaggle 表头变化）
COLS = {
    "anime_id": ["anime_id", "Anime ID", "MAL_ID", "ID"],
    "title": ["Name", "title", "Anime Title"],
    "title_en": ["English name", "English Name", "title_english"],
    "score": ["Score", "score"],
    "genres": ["Genres", "genres"],
    "themes": ["Themes", "themes"],
    "demographics": ["Demographics", "Demographic", "demographics"],
    "studios": ["Studios", "studios"],
    "an_type": ["Type", "type"],
    "episodes": ["Episodes", "episodes"],
    "aired": ["Aired", "aired"],
    "source": ["Source", "source"],
    "rating_age": ["Rating", "rating_age"],
    "popularity": ["Popularity", "popularity"],
    "synopsis": ["Synopsis", "sypnopsis", "synopsis"],  # 官方 CSV 表头历史上有拼写 sypnopsis
    "username": ["Username", "User Name", "username"],
    "rating": ["rating", "Rating", "score"],
}

BANNED_TYPES = {"hentai", "hent*ai", "erotica", "xxx"}
DEMOGRAPHIC_TAGS = {"shounen", "shoujo", "seinen", "josei", "kids"}

# 复用 Jikan 脚本的中英标签映射（缺失映射的英文标签原样保留）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from jikan_enrich import TAG_ZH
except ImportError:
    TAG_ZH = {}


def _zh(tag: str) -> str:
    return TAG_ZH.get(tag, tag)


def _pick(df: pd.DataFrame, field: str) -> str | None:
    for c in COLS[field]:
        if c in df.columns:
            return c
    return None


def _clean(v, default=""):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    s = str(v).strip()
    return default if s.upper() in ("UNKNOWN", "NA", "NAN", "") else s


def load_anime(anime_csv: str, conn: sqlite3.Connection, max_anime: int) -> int:
    """分块读作品表，过滤后写入。返回写入条数。"""
    use_cols = None
    n = 0
    for i, chunk in enumerate(pd.read_csv(anime_csv, chunksize=CHUNK)):
        if use_cols is None:  # 首块解析列名映射
            use_cols = {k: _pick(chunk, k) for k in COLS if _pick(chunk, k)}
            print(f"[anime] 列映射: {use_cols}")
            missing = [k for k in ("anime_id", "title", "score") if not use_cols.get(k)]
            if missing:
                sys.exit(f"[anime] 缺少必要列: {missing}")
        for _, row in chunk.iterrows():
            if n >= max_anime:
                break
            an_type = _clean(row[use_cols["an_type"]], "TV").lower()
            if an_type in BANNED_TYPES:          # 成人内容过滤
                continue
            score = _clean(row[use_cols["score"]], "0")
            try:
                score = float(score)
            except ValueError:
                continue                          # 无评分作品过滤
            if score <= 0:
                continue
            genres, themes, demo = [], [], []
            for g in _clean(row[use_cols["genres"]]).split(","):
                g = g.strip()
                if not g:
                    continue
                (demo if g.lower() in DEMOGRAPHIC_TAGS else genres).append(_zh(g))
            for t in _clean(row[use_cols.get("themes", "") or ""]).split(","):
                if t.strip():
                    themes.append(_zh(t.strip()))
            for t in _clean(row[use_cols["demographics"]]).split(",") if use_cols.get("demographics") else []:
                if t.strip():
                    demo.append(_zh(t.strip()))
            try:
                episodes = int(float(_clean(row[use_cols["episodes"]], "0")))
            except ValueError:
                episodes = 0
            try:
                pop = int(float(_clean(row[use_cols["popularity"]], "0")))
            except ValueError:
                pop = 0
            mal_id = int(row[use_cols["anime_id"]])
            title = _clean(row[use_cols["title"]])
            if not title:
                continue
            conn.execute(
                """INSERT OR REPLACE INTO anime (id,title,title_cn,an_type,episodes,aired,score,popularity,
                   studios,source,rating_age,genres,themes,demographics,franchise,characters,synopsis,mal_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mal_id, title, "", an_type.upper(), episodes,
                 _clean(row[use_cols["aired"]]), score, pop,
                 _clean(row[use_cols["studios"]]), _clean(row[use_cols["source"]]),
                 _clean(row[use_cols["rating_age"]]),
                 ",".join(dict.fromkeys(genres)), ",".join(dict.fromkeys(themes)),
                 ",".join(dict.fromkeys(demo)), "", "",
                 _clean(row[use_cols["synopsis"]])[:1500], mal_id),
            )
            n += 1
        if n % 2000 < CHUNK and i % 10 == 0:
            print(f"[anime] 已处理 {n} 部…")
        if n >= max_anime:
            break
    conn.commit()
    print(f"[anime] 完成，共写入 {n} 部（已过滤无评分与成人内容）")
    return n


def load_scores(score_csv: str, conn: sqlite3.Connection, limit: int) -> int:
    """分块读评分表：用户名映射为自增 id，仅保留库内作品。返回评分行数。"""
    use_cols = None
    user_ids: dict[str, int] = {}
    next_uid = 1 + (conn.execute("SELECT COALESCE(MAX(id),0) FROM users").fetchone()[0])
    n = 0
    valid_anime = {r[0] for r in conn.execute("SELECT id FROM anime")}
    for chunk in pd.read_csv(score_csv, chunksize=CHUNK):
        if use_cols is None:
            use_cols = {k: _pick(chunk, k) for k in ("username", "rating", "anime_id") if _pick(chunk, k)}
            print(f"[scores] 列映射: {use_cols}")
            if not use_cols.get("username") or not use_cols.get("rating"):
                sys.exit("[scores] 缺少 Username / rating 列")
        rows = []
        for _, r in chunk.iterrows():
            if limit and n >= limit:
                break
            rating = _clean(r[use_cols["rating"]])
            try:
                rating = float(rating)
            except ValueError:
                continue
            if not 1 <= rating <= 10:
                continue
            mal_id = int(r[use_cols["anime_id"]]) if use_cols.get("anime_id") else None
            if mal_id is None or mal_id not in valid_anime:
                continue
            uname = str(r[use_cols["username"]])
            uid = user_ids.get(uname)
            if uid is None:
                uid = next_uid
                user_ids[uname] = uid
                conn.execute("INSERT OR IGNORE INTO users (id,username) VALUES (?,?)", (uid, uname))
                next_uid += 1
            rows.append((uid, mal_id, rating))
            n += 1
        if rows:
            conn.executemany("INSERT OR REPLACE INTO ratings (user_id,anime_id,rating) VALUES (?,?,?)", rows)
            conn.commit()
        if limit and n >= limit:
            break
    print(f"[scores] 完成，共写入 {n} 条评分，{len(user_ids)} 个用户")
    return n


def main():
    ap = argparse.ArgumentParser(description="载入 Kaggle MAL 数据集到 SQLite")
    ap.add_argument("--anime", required=True, help="anime-dataset-2023.csv 路径")
    ap.add_argument("--scores", help="users-score-2023.csv 路径")
    ap.add_argument("--max-anime", type=int, default=12000, help="最多载入作品数")
    ap.add_argument("--limit-ratings", type=int, default=0, help="评分行数上限（0=不限）")
    args = ap.parse_args()

    # 建表（若目标库为空）：复用 database.init_db 的建表逻辑
    database.init_db()

    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("DELETE FROM anime")   # 真实数据替换种子数据
    conn.commit()

    load_anime(args.anime, conn, args.max_anime)
    if args.scores:
        load_scores(args.scores, conn, args.limit_ratings)
    conn.close()
    print("提示：cf.py 的相似度矩阵会在下次请求时自动重建；建议接着跑 jikan_enrich.py 补 themes/characters 缺口。")


if __name__ == "__main__":
    main()
