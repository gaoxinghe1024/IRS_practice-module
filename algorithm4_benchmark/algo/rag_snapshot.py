# -*- coding: utf-8 -*-
"""rag_snapshot.py — Page-4 (anime chat) retrieval pipeline, FROZEN COPY.

Source          : the system's backend/services/rag.py (AnimeFinder backend v1.4.1)
Copied          : 2026-09-30
Status          : frozen snapshot - do not edit by hand.
Allowed edits   : imports and data-file paths only, each marked `# SNAPSHOT-EDIT:`.
Verification    : tools/verify_snapshot.py compares every function against the
                  system copy (comments/whitespace stripped).
Note            : _llm_reply is copied verbatim for request-shape reference;
                  the evaluation replay never calls it.
"""
import os
import re
# SNAPSHOT-EDIT: was `from database import get_conn`. The system database module
# resolves anime.db next to the backend source tree; in this package the frozen
# knowledge base lives at <package_root>/data/kb/anime.db.
import sqlite3
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "kb", "anime.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # kept from system get_conn
    return conn

STOPWORDS = {"我", "想", "看", "一部", "一个", "的", "了", "是", "在", "有", "和", "与", "要",
             "找", "什么", "哪个", "哪部", "多少", "第", "集", "而且", "但是", "不要", "太",
             "最好", "类似", "像", "推荐", "吗", "呢", "啊", "这", "那个", "一部动漫"}

# LLM 配置（可选）：设置环境变量后自动切换为 LLM 生成
LLM_API_BASE = os.environ.get("LLM_API_BASE", "")   # 如 https://api.deepseek.com/v1
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")


def _tokenize(text: str) -> list[str]:
    """中文友好的轻量分词：CJK 拆单字 + 邻接二元组，拉丁词保持完整。

    演示级方案，无外部依赖；升级点：替换为 BGE 向量化语义检索。
    """
    text = text.lower()
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if ord(ch) > 0x2E80:  # CJK 及全角
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    tokens = [t for t in tokens if t and t not in STOPWORDS]
    # 邻接二元组：强化连续词匹配（如 夜神/对峙/巨人）
    cjk = [t for t in tokens if len(t) == 1 and ord(t) > 0x2E80]
    tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return [t for t in tokens if t not in STOPWORDS]


def _score(query_tokens: list[str], text: str) -> float:
    """词面重叠打分（演示级检索）。升级点：替换为向量余弦相似度。"""
    if not query_tokens:
        return 0.0
    text = text.lower()
    return sum(1 for t in query_tokens if t in text) / len(query_tokens)


def _extract_constraints(message: str) -> dict:
    """从自然语言中抽取硬约束（规则层：max_episodes / min_score / 排除关键词）。"""
    constraints = {}
    m = re.search(r"(\d+)\s*集", message)
    if m:
        constraints["max_episodes"] = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:分|分以上|分以上)", message)
    if not m:  # 「评分8.8以上 / 评分不低于8.8」语序
        m = re.search(r"评分\s*(?:在|不低于|不低于|大于|高于|超过)?\s*(\d+(?:\.\d+)?)", message)
    if m:
        constraints["min_score"] = float(m.group(1))
    for kw in ("短", "不要太长", "时间少"):
        if kw in message:
            constraints["max_episodes"] = constraints.get("max_episodes", 26)
    return constraints


def retrieve(message: str) -> dict:
    """RAG 检索层：返回 {'intent':..., 'anime':[...], 'scenes':[...], 'constraints':...}"""
    tokens = _tokenize(message)
    conn = get_conn()
    anime_rows = [dict(r) for r in conn.execute("SELECT * FROM anime").fetchall()]
    scene_rows = [dict(r) for r in conn.execute("SELECT * FROM scenes").fetchall()]
    conn.close()

    # 场景检索：名场面库（description+keywords）优先
    scene_hits = []
    for s in scene_rows:
        sc = _score(tokens, s["description"] + " " + s["keywords"] + " " + s["anime_title"])
        if sc > 0:
            scene_hits.append({**s, "score": sc})
    scene_hits.sort(key=lambda x: x["score"], reverse=True)

    # 作品检索：标题 + 标签 + 简介
    anime_hits = []
    for a in anime_rows:
        text = f"{a['title']} {a['title_cn']} {a['genres']} {a['themes']} {a['characters']} {a['synopsis']}"
        sc = _score(tokens, text)
        if sc > 0:
            anime_hits.append({**a, "ret_score": sc})
    anime_hits.sort(key=lambda x: (x["ret_score"], x["ret_score"] * x["popularity"]), reverse=True)

    # 意图判断：命中场景库 → 名场面匹配；否则 → 推荐意图
    intent = "scene" if scene_hits and scene_hits[0]["score"] >= 0.34 else "recommend"

    constraints = _extract_constraints(message)
    near_miss: list[dict] = []
    if intent == "recommend" and (constraints.get("max_episodes") or constraints.get("min_score")):
        def _apply(hits):
            out = hits
            if constraints.get("max_episodes"):
                out = [a for a in out if not a["episodes"] or a["episodes"] <= constraints["max_episodes"]]
            if constraints.get("min_score"):
                out = [a for a in out if a["score"] >= constraints["min_score"]]
            return out
        filtered = _apply(anime_hits)
        if not filtered:
            # 纯硬条件查询（如“评分8.8以上 20集以内”）：检索命中被硬条件淘汰时，用全目录兜底
            filtered = _apply([{**a, "ret_score": 0.0} for a in anime_rows])
        if not filtered:
            # 硬条件组合把目录筛空（如“9分以上 且 20集以内”）→ 找「最接近」的候选，
            # 而不是回一句“没理解”：恰好满足其中一个条件的作品，按另一条件的差距排序
            ms, me = constraints.get("min_score"), constraints.get("max_episodes")
            for a in anime_rows:
                ok_score = ms is None or a["score"] >= ms
                ok_ep = me is None or (not a["episodes"] or a["episodes"] <= me)
                if ok_score != ok_ep:
                    gap = (ms - a["score"]) if not ok_score else (a["episodes"] - (me or 0))
                    near_miss.append((max(0.0, gap), {**a, "ret_score": 0.0}))
            near_miss.sort(key=lambda t: t[0])
            filtered = [a for _, a in near_miss[:3]]
        anime_hits = filtered

    # “类似X”类查询：把消息里点名的作品本身挪到候选末尾（用户要的是"类似"，不是它）
    def _mentioned(a):
        return (a["title"] and a["title"].lower() in message.lower()) or \
               (a["title_cn"] and a["title_cn"] in message)
    if intent == "recommend" and len(anime_hits) > 1:
        mentioned = [a for a in anime_hits if _mentioned(a)]
        others = [a for a in anime_hits if not _mentioned(a)]
        if mentioned and others:
            anime_hits = others + mentioned

    return {"intent": intent, "anime": anime_hits[:5], "near_miss": near_miss[:3],
            "scenes": scene_hits[:3], "constraints": constraints, "tokens": tokens}


def _build_context(retrieved: dict) -> str:
    lines = []
    for a in retrieved["anime"][:3]:
        lines.append(f"《{a['title_cn'] or a['title']}》({a['title']}) 类型:{a['genres']} "
                     f"主题:{a['themes']} 集数:{a['episodes']} 评分:{a['score']} 简介:{a['synopsis'][:80]}")
    for s in retrieved["scenes"][:3]:
        lines.append(f"名场面[{s['anime_title']} 第{s['episode']}集]: {s['description']}")
    return "\n".join(lines)


def _template_reply(retrieved: dict) -> str:
    """无 LLM 时的模板生成，保证离线可演示。口吻偏聊天，区别于推荐位卡片。"""
    if retrieved["intent"] == "scene" and retrieved["scenes"]:
        s = retrieved["scenes"][0]
        return ("让我翻翻名场面库……找到啦！你说的应该是这一幕 👇")

    if not retrieved["anime"]:
        return ("这段描述我还没对上号，换个说法试试？比如：\n"
                "· “想找类似进击的巨人但不要太长的”\n"
                "· “L和夜神月对峙是哪一集”\n"
                "· “评分8.8以上 20集以内的心理战动漫”")

    cons = retrieved["constraints"]
    near = bool(retrieved.get("near_miss"))
    con_text = ""
    if cons.get("max_episodes"):
        con_text += f"「{cons['max_episodes']} 集以内」"
    if cons.get("min_score"):
        con_text += (" 且 " if con_text else "") + f"「评分 ≥ {cons['min_score']}」"

    if near:
        a = retrieved["anime"][0]
        reason = (f"它评分 {a['score']}" if cons.get("min_score") and a["score"] < cons["min_score"]
                  else f"它有 {a['episodes']} 集")
        names = [f"《{x['title_cn'] or x['title']}》" for x in retrieved["anime"][:3]]
        return (f"严格同时满足 {con_text} 的，知识库里暂时没有。\n"
                f"不过有很接近的：最推荐 **{names[0]}**——{reason}，就差一点点，"
                f"故事气质和你描述的很像。其他接近的还有 {'、'.join(names[1:])}。"
                f"\n要不要放宽一个条件再试试？")

    a = retrieved["anime"][0]
    names = [f"《{x['title_cn'] or x['title']}》" for x in retrieved["anime"][:3]]
    return (f"按你说的条件 {con_text}，我聊下来最想推给你的是 **{names[0]}**："
            f"{a['synopsis'][:60]}……评分 {a['score']}，{a['episodes']} 集。"
            + (f"\n\n其他候选：{'、'.join(names[1:])}" if len(names) > 1 else "")
            + "\n\n感兴趣的话，点下面的卡片就能看详情～")


def _llm_reply(retrieved: dict, message: str) -> str | None:
    """配置了 LLM API 时调用 OpenAI 兼容接口生成回复；失败返回 None 回退模板。"""
    if not (LLM_API_BASE and LLM_API_KEY):
        return None
    import json as _json
    import urllib.request
    context = _build_context(retrieved)
    prompt = (
        "你是一个懂动漫的推荐助手，请根据下面的知识库内容用中文自然地回答用户，"
        "不要编造知识库里没有的作品。\n\n知识库：\n" + context + "\n\n用户：" + message
    )
    body = _json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(
        LLM_API_BASE.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def chat(message: str) -> dict:
    """RAG 主流程：检索 → 上下文拼装 → 生成（LLM 或模板）。

    命中名场面时额外返回 scene_focus（结构化「回忆片段」），
    前端用它渲染对话式的场景气泡，与推荐位卡片刻意区分。
    """
    retrieved = retrieve(message)
    reply = _llm_reply(retrieved, message) or _template_reply(retrieved)
    src = "llm" if (LLM_API_BASE and LLM_API_KEY and reply) else "template"

    scene_focus = None
    if retrieved["intent"] == "scene" and retrieved["scenes"]:
        s = retrieved["scenes"][0]
        conn = get_conn()
        row = conn.execute("SELECT id, title, title_cn, image_url FROM anime WHERE title LIKE ?",
                           (f"%{s['anime_title']}%",)).fetchone()
        conn.close()
        scene_focus = {"anime_id": row["id"] if row else None,
                       "title": s["anime_title"],
                       "title_cn": (row["title_cn"] if row else ""),
                       "image_url": (row["image_url"] if row else ""),
                       "episode": s["episode"],
                       "description": s["description"],
                       "keywords": [k for k in s["keywords"].split(",") if k]}

    return {
        "reply": reply,
        "source": src,
        "scene_focus": scene_focus,
        "retrieved": {
            "anime": [{k: v for k, v in a.items() if k in ("id", "title", "title_cn", "genres", "themes", "episodes", "score")}
                      for a in retrieved["anime"]],
            "scenes": [{"anime_title": s["anime_title"], "episode": s["episode"],
                        "description": s["description"]} for s in retrieved["scenes"]],
        },
    }
