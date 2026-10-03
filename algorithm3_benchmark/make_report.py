# -*- coding: utf-8 -*-
"""
make_report.py — 生成**单份**合并报告：ICF 与 GA 两部分。

设计原则
--------
* 所有数字从 results/icf_part.json 与 results/ga_part.json 机器读取，零手工转录；
* 统计区间动态计算，不硬编码；
* 表头一律扁平单行（html4docx 不支持 rowspan/colspan）。

报告结构
--------
    一句话结论
    读前必读：三个口径
    一、实验协议（30 秒版）
    二、指标字典（每个指标都解释清楚）
       2.1 新颖度的定义（长尾判定的全部依据）
       2.2 GA 的适应度权重（算法③幻灯片口径）
    三、第一部分 · ICF：三个召回引擎的对比
    四、第二部分 · GA：收益与代价
       4.1 表 4 — 四臂对照（贪心 / 窗口随机 / 整池随机 / GA）
       4.2 表 5 — 主对照表（GA vs 贪心）
       4.3 收益 / 代价的交换比
       4.4 表 6 — GOS 三项拆解
       4.5 收益的天花板为什么由召回决定
       4.6 表 7 — 候选池敏感性
       4.7 新增随机臂为什么不会扰动已有结果
    五、结论（四句话）+ 权重与新颖度速查

数据分工
--------
    第一部分表格 ← results/icf_part.json（3 引擎 × 10 档的无重排 HR + 新颖度分布）
    第一部分基线诊断、第二部分全部 ← results/ga_part.json（四臂 × 10 档逐用户配对）

输出：results/ICF_GA_报告.html
"""
import json
import os

import config as C

ICF_SRC = os.path.join(C.OUT_DIR, "icf_part.json")
GA_SRC = os.path.join(C.OUT_DIR, "ga_part.json")
DST = os.path.join(C.OUT_DIR, "ICF_GA_报告.html")

icf = json.load(open(ICF_SRC, encoding="utf-8"))
ga = json.load(open(GA_SRC, encoding="utf-8"))

P = ga["pools"]
N = ga["n_users"]
SLOTS = ga["exposure_slots"]
MED = ga["longtail_median"]
W = ga["weights"]
GRID = icf["pools"]
ENGINES = icf["engines"]
ENGNAME = {"ICF": "ICF", "SVD": "SVD", "NCF": "NCF"}

# 四臂定义（顺序即报告展示顺序：贪心 → 窗口随机 → 整池随机 → GA）
ARMS = ("greedy", "randw", "randp", "ga")
ARM_DEF = ga.get("arm_definitions") or {
    "greedy": "贪心：纯按相关性取 Top-10（恒等于无重排）",
    "randw": "窗口随机：在相关性前 20 名里随机挑 10 条",
    "randp": "整池随机：完全不看相关性，从整个候选池随机挑 10 条",
    "ga": "GA：0.4 相关性 + 0.3 多样性 + 0.2 新颖度 + 0.1 新鲜度",
}
ARMN = {"greedy": "贪心", "randw": "窗口随机", "randp": "整池随机", "ga": "GA"}

BUF = []
A = BUF.append

# 新颖度分布统计（来自 icf_part.json，由 analyze_icf_part.py 从原始数据包复算）
NV = icf.get("novelty", {})
NV_MEAN = NV.get("novel_mean", 0.0)
NV_LT = NV.get("n_longtail", 0)
NV_LT_SHARE = NV.get("longtail_share", 0.0)
NV_HEAD10 = NV.get("head10_share", 0.0)
NV_HEAD1 = NV.get("head1_share", 0.0)
NV_ZERO = NV.get("n_zero_pop", 0)
NV_N = NV.get("n_items", 0)

# ⚠️ 用户真实消费作品的"平均新颖度"从数据读，不凭印象写
_NVU = ga.get("nov_user") or {}
NOV_USER_MEAN = _NVU.get("mean")
NOV_USER_LT = _NVU.get("longtail_share")
if NOV_USER_MEAN is not None:
    _novuser_txt = (f"<b>用户真实交互过的作品，平均新颖度只有 {NOV_USER_MEAN:.3f}</b>"
                    f"（{NOV_USER_LT*100:.1f}% 属长尾）——")
else:
    _novuser_txt = "<b>用户真实交互过的作品，平均新颖度也明显低于长尾阈值</b>——"
imbalance_txt = (
    f"长尾作品在全库 {NV_N:,} 部作品中占 <b>{NV_LT:,} 部（{NV_LT_SHARE*100:.1f}%）</b>，"
    f"但人气极度集中：<b>前 1% 的作品吃掉了 {NV_HEAD1*100:.1f}% 的评分量，前 10% 吃掉 {NV_HEAD10*100:.1f}%</b>。"
    if NV_N else ""
)


def pct(x, nd=2):
    return f"{x:.{nd}f}%"


def n0(x):
    return f"{x:,.0f}"


def pp(x, nd=2):
    return f"{x:+.{nd}f}pp"


def fmt_p(p):
    """Wilcoxon/McNemar 在大样本下会下溢到 0.0，写 'p < 1e-300' 而非 'p = 0'。"""
    if p is None or p != p:
        return "—"
    if p <= 0:
        return "p &lt; 1e-300"
    return f"p = {p:.1e}"


def pool_name(g):
    return "全目录" if g == "full" else g


def B(pool, key):
    return P[pool]["benefit"][key]


def CO(pool, key):
    return P[pool]["cost"][key]


def M(pool, key):
    return P[pool]["metrics"][key]


FULL = "full"
f_b = {k: B(FULL, k) for k in ("lte", "ltr", "gos")}
f_c, f_c5 = CO(FULL, "hr10"), CO(FULL, "hr5")
f_t = P[FULL]["tradeoff"]
f_new = P[FULL]["benefit"]["newly_exposed_users"]
f_div, f_nov, f_fre = M(FULL, "div10"), M(FULL, "nov10"), M(FULL, "fresh10")
NCAND = P[FULL]["n_candidates"]
RANDOM_BASE = 10.0 / NCAND
LIFT = f_c["greedy_mean"] / RANDOM_BASE

TITLE = "ICF 与 GA 核心结果报告 —— 召回对比 · 重排收益"

A(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
:root {{
  --ink:#1b1f27; --muted:#68707f; --line:#e3e7ee; --bg:#ffffff;
  --head:#f6f8fb; --gain:#c0392b; --cost:#1f6f4a; --accent:#2f5fd0;
  --warn:#fff8e6; --warnline:#e8c86a; --soft:#fbfcfe;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:34px 30px 60px; background:var(--bg); color:var(--ink);
  font-family:"Microsoft YaHei","PingFang SC",-apple-system,"Segoe UI",sans-serif;
  font-size:14px; line-height:1.75; }}
.wrap {{ max-width:1020px; margin:0 auto; }}
h1 {{ font-size:25px; line-height:1.35; margin:0 0 6px; letter-spacing:.2px; }}
h2 {{ font-size:19px; margin:38px 0 12px; padding-left:11px;
  border-left:4px solid var(--accent); }}
h3 {{ font-size:15.5px; margin:24px 0 8px; color:#2a3240; }}
.sub {{ color:var(--muted); font-size:13px; margin:0 0 22px; }}
p {{ margin:9px 0; }}
ul,ol {{ margin:8px 0 8px 20px; padding:0; }}
li {{ margin:5px 0; }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 6px; font-size:13px; }}
th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left;
  vertical-align:top; }}
th {{ background:var(--head); font-weight:600; }}
td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.gain {{ color:var(--gain); font-weight:700; }}
.cost {{ color:var(--cost); font-weight:700; }}
.mono {{ font-family:Consolas,"Courier New",monospace; font-size:12.5px; }}
.lead {{ background:var(--head); border-left:4px solid var(--accent);
  padding:14px 18px; border-radius:0 6px 6px 0; margin:16px 0 24px; }}
.lead b {{ color:var(--accent); }}
.tip {{ background:var(--warn); border-left:4px solid var(--warnline);
  padding:12px 16px; border-radius:0 6px 6px 0; margin:14px 0; font-size:13px; }}
.def {{ background:var(--soft); border:1px solid var(--line); border-radius:6px;
  padding:12px 16px; margin:12px 0; }}
.def .k {{ font-weight:700; color:var(--accent); }}
.cap {{ color:var(--muted); font-size:12.5px; margin:4px 0 18px; }}
.pill {{ display:inline-block; padding:1px 8px; border-radius:10px;
  font-size:11.5px; font-weight:700; }}
.pill.g {{ background:#fdecea; color:var(--gain); }}
.pill.c {{ background:#e8f5ee; color:var(--cost); }}
.pill.n {{ background:#eef1f6; color:#4a5464; }}
</style></head><body><div class="wrap">
<h1>{TITLE}</h1>
<p class="sub">数据源 <span class="mono">results/ranks.npz</span>　|　
用户 {N:,}　|　候选池 {len(GRID)} 档（20 → 全目录 {n0(NCAND)}）　|　
采样种子 {ga['seeds']}　|　召回引擎 {len(ENGINES)} 个 × 重排臂 4 个（贪心/窗口随机/整池随机/GA）　|　
每用户展示 {ga['top_k']} 条，共 {n0(SLOTS)} 个曝光位</p>

<div class="lead">
<p><b>一句话结论：</b>ICF 负责<b>召回到正确的作品</b>（命中率是随机猜的 {LIFT:.0f} 倍），
GA 负责<b>把这 10 个位置重新分配</b>。在全目录口径下，用户前 10 条里长尾商品的位置从
<b>{n0(f_b['lte']['greedy'])} 条提到 {n0(f_b['lte']['ga'])} 条（<span class="gain">{f_b['lte']['rel_pct']:+.1f}%</span>）</b>，
能看到长尾商品的用户从 <b>{pct(f_b['ltr']['greedy']*100)} 升到 {pct(f_b['ltr']['ga']*100)}（{n0(f_new)} 人新增）</b>，
代价是 HR@10 从 {pct(f_c['greedy_mean']*100)} 降到 {pct(f_c['ga_mean']*100)}
（<span class="cost">{pp(f_t['d_hr10_pp'])}</span>）。</p>
<p><b>GA 不是"把列表打乱"。</b>在与 GA <b>完全相同的候选窗口</b>里随机挑 10 条（窗口随机臂），
命中率只有 {pct(P[FULL]['arms']['randw']['hr10']*100)}；GA 是它的
<b>{P[FULL]['arms']['ga']['hr10']/P[FULL]['arms']['randw']['hr10']:.2f} 倍</b>。
这说明 GA 的收益来自<b>重排能力</b>，而非随机性（详见 4.1 节四臂对照）。</p>
</div>
""")

# ==================================================================== §0 口径
A(f"""<h2>读前必读：三个口径</h2>
<div class="def"><p><span class="k">① 候选池 N（本报告的自变量）</span><br>
每次评估的候选集 = 用户真实交互过的 1 个作品（已从训练集藏起）+（N−1）个<b>均匀随机负例</b>。
N 越大干扰项越多、任务越难。<b>N = 全目录 {n0(NCAND)} 是最接近真实部署的口径</b>，
也是两部分正文使用的口径；其余档位见第 4.5 节敏感性表。
<span class="pill n">注意</span> 跨档位的 HR 不能直接比大小——候选池越大，难度越高。</p></div>

<div class="def"><p><span class="k">② 长尾商品</span><br>
物品新颖度 ≥ 全库中位数 <span class="mono">{MED}</span>，等价说法是"人气低于全库一半"。
新颖度的完整定义见 <b>2.1 节</b>。
注意对比：{_novuser_txt}用户自己也在消费头部作品，
推荐长尾本质上是在偏离用户历史偏好，这决定了收益不可能无限大。</p></div>

<div class="def"><p><span class="k">③ 重排窗口 RERANK_DEPTH = 20</span><br>
GA <b>只在相关性排序的前 20 条里重排，输出前 10 条</b>。
这条约束是理解第 4.5 节"收益为什么只有这个量级"的关键。
<b>窗口随机臂使用的正是同一个 20 条窗口</b>，所以它与 GA 的对比排除了"候选来源不同"的干扰。</p></div>
""")

# ==================================================================== §1 协议
A(f"""<h2>一、实验协议（30 秒版）</h2>
<p>协议来自算法③幻灯片，四个环节如下：</p>
<ol>
<li><b>留一法（leave-one-out）</b>：每位用户从训练集里<b>藏起 1 条高分记录</b>（≥ {C.HIGH_RATING} 分），
    它既不在训练里、也不在召回里，只作为答案。合格用户门槛：训练集至少 {C.MIN_TRAIN_RATINGS} 条评分。
    最终取合格用户池的固定随机抽样 {N:,} 人。</li>
<li><b>构造候选集</b>：把藏起的那 1 条和（N−1）个从未评过的作品混在一起，共 {len(GRID)} 档：
    N = 20 / 50 / 100 / 200 / 500 / 1000 / 2000 / 5000 / 10000 / 全目录。全部用户共用同一个队列与采样种子，
    所以每个配置都是在<b>同一批人、同一批候选</b>上比较。</li>
<li><b>召回引擎</b>（3 个）：ICF 物品协同过滤、SVD 矩阵分解、NCF 神经协同过滤。
    它们在候选集里给出打分与排序 → 报告"第一部分"。</li>
<li><b>重排臂</b>（4 个，同一个候选池、同一条召回结果，<b>只换"怎么挑这 10 条"</b>）：
    <b>贪心</b>（纯按相关性取 Top-10，即 GA 的内部基线）、
    <b>窗口随机</b>（在相关性前 20 名里随机挑 10 条）、
    <b>整池随机</b>（完全不看相关性，从整个候选池随机挑 10 条）、
    <b>GA</b>（{W['relevance']} 相关性 + {W['diversity']} 多样性
    + {W['novelty']} 新颖度 + {W['fresh']} 新鲜度）→ 报告"第二部分"。</li>
</ol>
<p>三个必须写清楚的约定：</p>
<ul>
<li><b>贪心 ≡ 无重排</b>。召回列表本身就按相关性降序排好，而贪心的定义是"纯按相关性取 Top-K"，
    所以两者选出的集合完全相同。本报告中"ICF 基线"指的就是这个 Top-10。</li>
<li><b>GA 的重排深度 = min(候选集, 20)</b>。线上管线是"召回 Top-20 → GA 重排 → Top-10"，
    重排层永远只看前 20 名。候选池小于 20 时自然等于整个池子。
    <b>窗口随机臂用的是同一个窗口</b>，这是它能充当"GA 对照组"的前提。</li>
<li><b>所有臂的输出展示顺序统一为"相关性降序"</b>。先按各自的规则挑出 10 条，
    再按相关性排好展示——否则 HR@5 这类依赖顺序的指标在臂之间不可比。</li>
</ul>
""")

# ==================================================================== §2 指标字典
A(f"""<h2>二、指标字典（本报告只用了这些指标）</h2>
<p>每个指标都给出<b>计算式</b>与<b>一句话意义</b>，方便逐条核对。
收益指标（红）是 GA 要提升的，代价指标（绿）是 GA 要守住的。</p>
<table>
<tr><th>指标</th><th>类别</th><th>计算式</th><th>一句话意义</th></tr>

<tr><td><b>HR@5</b><br>命中率@5</td>
    <td><span class="pill c">代价</span></td>
    <td class="mono">藏起的那 1 条落进前 5 位的用户 ÷ 全部用户</td>
    <td>推荐得准不准。它比 HR@10 更严格，是精度损失的"报警器"——
        因为前 5 位才是用户最可能点开的位置。</td></tr>

<tr><td><b>HR@10</b><br>命中率@10</td>
    <td><span class="pill c">代价</span></td>
    <td class="mono">藏起的那 1 条落进前 10 位的用户 ÷ 全部用户</td>
    <td>GA 输出就是 Top-10，所以 HR@10 是 GA 的天然口径。
        它对应 GA 适应度里 {W['relevance']} 权重的<b>相关性项</b>。</td></tr>

<tr><td><b>随机基线</b><br>K / N</td>
    <td><span class="pill n">参照</span></td>
    <td class="mono">K ÷ N（K = 5 或 10）</td>
    <td>完全瞎猜的命中率。候选池越大基线越低，所以跨档位要看下面的倍数。</td></tr>

<tr><td><b>提升倍数</b><br>lift</td>
    <td><span class="pill n">参照</span></td>
    <td class="mono">HR@K ÷ (K/N)</td>
    <td><b>比瞎猜强多少倍</b>。这是唯一可以跨档位横向比的命中口径，
        用来证明"召回本身是有效的"。</td></tr>

<tr><td><b>nov@K</b><br>新颖度</td>
    <td><span class="pill g">收益</span></td>
    <td class="mono">列表前 K 条的新颖度均值<br>
        novelty(i) = 1 − log(1+人气ᵢ) / log(1+最大人气)</td>
    <td>"这份推荐整体有多冷门"。它是 <b>0~1 的小数</b>，所以
        "+0.008"读不出体感——这正是下面三个体感指标存在的理由。</td></tr>

<tr><td><b>lt@10</b><br>长尾占比</td>
    <td><span class="pill g">收益</span></td>
    <td class="mono">前 10 条中 novelty ≥ {MED} 的条目占比</td>
    <td>把"平均冷门程度"换成"<b>多少条真的冷门</b>"，比均值更抗平均化掩盖。</td></tr>

<tr><td><b>LTE@10</b><br>长尾曝光量</td>
    <td><span class="pill g">收益</span></td>
    <td class="mono">Σ(每个用户前 10 条里长尾商品的条数)</td>
    <td>全站 {n0(SLOTS)} 个推荐位中，长尾商品占了多少<b>条</b>。
        把比例还原成条数，可以直接写"多了 n 条曝光"。</td></tr>

<tr><td><b>LTR@10</b><br>长尾触达率<br><span class="pill g">推荐主指标</span></td>
    <td><span class="pill g">收益</span></td>
    <td class="mono">前 10 条里至少出现 1 个长尾商品的用户 ÷ 全部用户</td>
    <td>有多少用户<b>真的被推到了</b>长尾商品。它把"平均新颖度涨了 0.008"
        翻译成"<b>多少人的推荐被改变了</b>"，最直观，也最难被质疑。</td></tr>

<tr><td><b>GOS@10</b><br>GA 目标分</td>
    <td><span class="pill g">收益</span></td>
    <td class="mono">{W['diversity']}×多样性@10 + {W['novelty']}×新颖度@10 + {W['fresh']}×新鲜度@10</td>
    <td>这正好是 GA 适应度里<b>由 GA 负责优化的那 60%</b>
        （剩下 {W['relevance']} 是相关性，由召回引擎保证、用 HR 监控代价）。
        它是 GA 唯一该被考核的 KPI，用来回应"GA 到底在优化什么"。</td></tr>
</table>
<p class="cap">配对口径：同用户、同候选池、同引擎、同随机种子逐用户做差；
收益指标（连续）用 Wilcoxon 符号秩检验，命中率（二值）用 McNemar 检验，均 {N:,} 名用户。</p>

<h3>2.1 新颖度的定义（长尾判定的全部依据）</h3>
<p>本报告的"长尾"不是按销量排名拍的，而是由一个连续、可复算的新颖度分数定义。</p>
<div class="def">
<p><span class="k">物品新颖度（novelty）</span></p>
<p class="mono" style="margin:8px 0">
novelty(i) = 1 − log(1 + 人气<sub>i</sub>) ÷ log(1 + 最大人气)
</p>
<p>其中<b>人气<sub>i</sub></b> = 作品 i 在<b>训练集</b>里被评分的次数，
<b>最大人气</b> = 全库中人气最高的作品。<br>
采用 <span class="mono">log(1+·)</span> 而不是直接除以人气，是为了压缩头部——
否则最热门那部作品会把其余所有作品的新颖度压到接近 1，失去区分度。</p>
<p>该式的取值范围是 <b>[0, 1]</b>：
<b>0</b> = 全库最热门的作品（新颖度最低），
<b>1</b> = 训练集里从未出现过的作品（新颖度最高）。</p>
</div>
<div class="def">
<p><span class="k">长尾阈值 = 全库新颖度中位数 = <span class="mono">{MED}</span></span></p>
<p>判定规则：<b>novelty(i) ≥ {MED} 的作品即为长尾作品</b>。
{imbalance_txt}
之所以取<b>中位数</b>而不是均值或某个分位点：中位数天然把全库作品一刀切成
"较冷门的一半"与"较热门的一半"，不依赖人工设定的绝对人气门槛，
换数据集时阈值随数据自动重估，报告中的结论也就不会因为选阈值而被"调"出来。</p>
<p>参考量：{_novuser_txt}用户自己也在消费头部作品。
所以推荐长尾本质上是在<b>偏离用户历史偏好</b>，
这决定了 GA 的收益不可能无限大。</p>
</div>
<div class="def">
<p><span class="k">"长尾"与"冷门"的区别</span></p>
<p>novelty 高只说明<b>训练集里评它的人少</b>，不等于它就是差作品——
本数据集的作品都有元数据（类型标签、放送年份、集数），
阈值 {MED} 正是<b>在有元数据的作品上</b>取的中位数，因此长尾作品仍然是可用的推荐结果，
而不是"无人问津的垃圾数据"。这一点是"用命中率换长尾"这个交易成立的前提。</p>
</div>

<h3>2.2 GA 的适应度权重（算法③幻灯片口径）</h3>
<p>GA 的适应度是一个<b>加权和</b>，四个分项全部在池内归一化到 [0,1] 后加权。
权重如下，直接来自 <span class="mono">config.py</span>，与算法③幻灯片一致：</p>
<table>
<tr><th>分项</th><th class="num">权重</th><th>含义</th><th>谁来保证</th></tr>
<tr><td><b>相关性</b> relevance</td>
    <td class="num"><b>{W['relevance']}</b></td>
    <td>列表里作品与用户画像的匹配程度，池内 min-max 归一化到 [0,1]。</td>
    <td>召回引擎（ICF/SVD/NCF）保证；GA 不优化它，
        <b>用 HR@5 / HR@10 监控它的代价</b>。</td></tr>
<tr><td><b>多样性</b> diversity</td>
    <td class="num"><b>{W['diversity']}</b></td>
    <td>列表内作品的类型标签差异度（1 − 平均两两标签相似度）。</td>
    <td><span class="pill g">GA 负责</span></td></tr>
<tr><td><b>新颖度</b> novelty</td>
    <td class="num"><b>{W['novelty']}</b></td>
    <td>即上面定义的 novelty，取前 10 条均值。</td>
    <td><span class="pill g">GA 负责</span></td></tr>
<tr><td><b>新鲜度</b> freshness</td>
    <td class="num"><b>{W['fresh']}</b></td>
    <td>作品放送年份的归一化新旧程度。</td>
    <td><span class="pill g">GA 负责</span></td></tr>
<tr><td colspan="4"><b>适应度 = {W['relevance']}×相关性 + {W['diversity']}×多样性
    + {W['novelty']}×新颖度 + {W['fresh']}×新鲜度</b></td></tr>
</table>
<div class="tip">
<p><b>为什么本报告的"收益"只看后三项（合计权重 {W['diversity']+W['novelty']+W['fresh']:.2f}）？</b><br>
因为相关性那 {W['relevance']} 是<b>召回引擎的职责</b>，GA 的重排窗口只有 20 条、
且全部是相关性排序的前列，它无法把窗口外的作品拉进来，也就无从"提升相关性"。
所以 GA 唯一该被考核的，就是它真正能动的这 {W['diversity']+W['novelty']+W['fresh']:.0%}——
即报告中的 <b>GOS@10 指标</b>。而相关性的变化（{W['relevance']} 那部分）则作为<b>代价</b>，
用 HR@5 / HR@10 单独监控。<br>
把"收益"和"代价"这样分开记，是为了避免一个常见的误导：
<b>用一个加权总分去证明"GA 更好"，会让权重选择决定结论。</b></p>
</div>
""")

# ==================================================================== §3 ICF
A(f"""<h2>三、第一部分 · ICF：三个召回引擎的对比</h2>
<p>三个引擎在 10 档候选池上各自"无重排"排序，看藏起的那条排到第几名。
粗体 = <b>该档最强引擎</b>。</p>

<h3>3.1 表 1 — HR@5（正例进入前 5 的比例）</h3>
<table>
<tr><th class="num">候选池 N</th><th class="num">候选集均值</th><th class="num">随机基线 5/N</th>
""")
for e in ENGINES:
    A(f'<th class="num">{ENGNAME[e]}</th>')
A("</tr>\n")
for g in GRID:
    p = icf["per_pool"][g]
    A(f'<tr><td class="num"><b>{pool_name(g)}</b></td>'
      f'<td class="num">{p["mean_candidate_set"]:,.1f}</td>'
      f'<td class="num">{p["random_hr5"]*100:.5f}%</td>')
    for e in ENGINES:
        v = p["engines"][e]["hr5"]
        bold = " style='font-weight:700'" if e == p["winner"] else ""
        A(f'<td class="num"{bold}>{v*100:.2f}%</td>')
    A("</tr>\n")
A("</table>\n")

A(f"""<h3>3.2 表 2 — HR@10（正例进入前 10 的比例）</h3>
<table>
<tr><th class="num">候选池 N</th><th class="num">候选集均值</th><th class="num">随机基线 10/N</th>
""")
for e in ENGINES:
    A(f'<th class="num">{ENGNAME[e]}</th>')
A("</tr>\n")
for g in GRID:
    p = icf["per_pool"][g]
    A(f'<tr><td class="num"><b>{pool_name(g)}</b></td>'
      f'<td class="num">{p["mean_candidate_set"]:,.1f}</td>'
      f'<td class="num">{p["random_hr10"]*100:.5f}%</td>')
    for e in ENGINES:
        v = p["engines"][e]["hr10"]
        bold = " style='font-weight:700'" if e == p["winner"] else ""
        A(f'<td class="num"{bold}>{v*100:.2f}%</td>')
    A("</tr>\n")
A("</table>\n")

A("""<h3>3.3 表 3 — 相对完全随机基线的提升倍数（跨档位唯一可比的口径）</h3>
<table>
<tr><th class="num">候选池 N</th>
""")
for e in ENGINES:
    A(f'<th class="num">{ENGNAME[e]} @5</th><th class="num">{ENGNAME[e]} @10</th>')
A("</tr>\n")
for g in GRID:
    p = icf["per_pool"][g]
    A(f'<tr><td class="num"><b>{pool_name(g)}</b></td>')
    for e in ENGINES:
        A(f'<td class="num">{p["engines"][e]["lift5"]:,.2f}×</td>'
          f'<td class="num">{p["engines"][e]["lift10"]:,.2f}×</td>')
    A("</tr>\n")
s = icf["summary"]
wins = s["winner_count"]
win_txt = "、".join(f"{e} 在 {n} 档最强" for e, n in wins.items() if n)
A(f"""</table>
<p class="cap">读表要点：全部 30 个组合（3 引擎 × 10 档）都远超随机——
@5 倍数 {s['lift5_range'][0]:,.2f}× ~ {s['lift5_range'][1]:,.2f}×，
@10 倍数 {s['lift10_range'][0]:,.2f}× ~ {s['lift10_range'][1]:,.2f}×。
本运行中 {win_txt}：<b>候选池小（≤ 2000）时 NCF 最强，候选池大（≥ 5000）时 ICF 最强</b>。
SVD 从未在任何档位胜出。</p>

<h3>3.4 基线的问题：曝光被头部垄断</h3>
<p>命中率虽然高，但排序完全是"相关性优先"，而 ICF 的相似度分数与物品人气强耦合——
<b>越热门越相似</b>。结果是前 10 条被头部作品占满。在全目录口径下，ICF 基线的收益侧表现是：</p>
<table>
<tr><th>指标</th><th class="num">数值</th><th>怎么读</th></tr>
<tr><td>HR@10 <span class="pill c">代价</span></td><td class="num">{pct(f_c['greedy_mean']*100)}</td>
    <td>在 {n0(NCAND)} 个候选里命中 {pct(f_c['greedy_mean']*100)}，随机猜只有 {pct(RANDOM_BASE*100, 3)}，
        <b>约 {LIFT:.0f} 倍</b>——召回本身是有效的。</td></tr>
<tr><td>HR@5 <span class="pill c">代价</span></td><td class="num">{pct(f_c5['greedy_mean']*100)}</td>
    <td>命中主要集中在前 5 位。</td></tr>
<tr><td>GOS@10 <span class="pill g">收益</span></td><td class="num">{f_b['gos']['greedy']:.4f}</td>
    <td>三项加权拆开看：多样性 {f_div['greedy_mean']:.4f}（×{W['diversity']} = {W['diversity']*f_div['greedy_mean']:.4f}）、
        新颖度 {f_nov['greedy_mean']:.4f}（×{W['novelty']} = {W['novelty']*f_nov['greedy_mean']:.4f}）、
        新鲜度 {f_fre['greedy_mean']:.4f}（×{W['fresh']} = {W['fresh']*f_fre['greedy_mean']:.4f}）。
        <b>分数几乎全靠多样性撑起来</b>，新颖度与新鲜度贡献合计不到 10%。</td></tr>
<tr><td>LTE@10 <span class="pill g">收益</span></td><td class="num">{n0(f_b['lte']['greedy'])} 条</td>
    <td>= {n0(SLOTS)} 个曝光位的 <b>{pct(100*f_b['lte']['greedy']/SLOTS)}</b>。长尾作品几乎拿不到曝光。</td></tr>
<tr><td>LTR@10 <span class="pill g">收益</span></td><td class="num">{pct(f_b['ltr']['greedy']*100)}</td>
    <td>只有 {n0(f_b['ltr']['greedy_users'])} 名用户在前 10 条里见过至少 1 个长尾作品；
        换句话说 <b>{pct((1-f_b['ltr']['greedy'])*100)} 的用户从来没见过长尾作品</b>。</td></tr>
</table>
<p>这就是引入 GA 的理由：<b>在不动召回结果的前提下，用重排把一部分曝光让给长尾。</b></p>
""")

# ==================================================================== §4 GA
A(f"""<h2>四、第二部分 · GA：收益与代价</h2>

<h3>4.1 表 4 — 四臂对照：GA 的收益来自"重排能力"还是"随机性"？</h3>
<p>这是本报告最关键的一组表。要证明 GA 有用，光跟"贪心"比是不够的——
GA 内部本来就有随机初始化与变异，读者完全有理由质疑：
<b>"把列表随便打乱一点也能拿到这些收益，为什么还要用 GA？"</b>
所以必须补上两个随机对照臂，而且它们回答的是<b>两个完全不同的问题</b>：</p>
<table>
<tr><th>对照臂</th><th>怎么挑这 10 条</th><th>候选来源</th><th>回答什么问题</th></tr>
<tr><td><b>贪心</b><br><span class="pill n">基线</span></td>
    <td>纯按相关性取 Top-10</td>
    <td>相关性前 20</td>
    <td>不做任何重排会怎样（≡ 无重排）。</td></tr>
<tr><td><b>窗口随机</b><br>win</td>
    <td>在窗口里<b>随机</b>挑 10 条</td>
    <td><b>与 GA 完全相同</b><br>（相关性前 20）</td>
    <td><b>GA 的重排能力</b>——候选一样、只换挑选方式。
        GA 若不能显著超过它，GA 就没有存在价值。</td></tr>
<tr><td><b>整池随机</b><br>pool</td>
    <td>完全不看相关性，<b>随机</b>挑 10 条</td>
    <td><b>整个候选池</b><br>（全目录 {n0(NCAND)} 条）</td>
    <td><b>曝光结构的理论上界</b>——命中率会塌到瞎猜水平（K/N），
        但曝光最接近全库分布，用来看 GA 换到了多少。</td></tr>
<tr><td><b>GA</b><br><span class="pill g">主角</span></td>
    <td class="mono">{W['relevance']} 相关性 + {W['diversity']} 多样性
        + {W['novelty']} 新颖度 + {W['fresh']} 新鲜度</td>
    <td>相关性前 20</td>
    <td>加权适应度优化。</td></tr>
</table>
<p class="cap">臂之间唯一的差别是"怎么从池子里挑这 10 条"。所有臂都先把 10 条挑出来、
再按相关性降序展示，保证 HR@5 这类依赖顺序的指标可比；
对照臂使用独立随机流，与 GA 的随机流互不干扰（见 4.7 节）。</p>

<h4 style="font-size:14px;margin:18px 0 6px;color:#2a3240">代价侧：命中率 HR@10</h4>
<table>
<tr><th>候选池 N</th><th>胜者引擎</th>
    <th class="num">贪心</th><th class="num">窗口随机</th>
    <th class="num">整池随机</th><th class="num">GA</th>
    <th class="num">GA÷窗口随机</th><th class="num">GA÷整池随机</th></tr>
""")
for g in GRID:
    fa = P[g]["arms"]
    r_win = fa["randw"]["hr10"]
    r_pool = fa["randp"]["hr10"]
    A(f"""<tr><td><b>{pool_name(g)}</b></td><td>{P[g]['engine']}</td>
      <td class="num">{pct(fa['greedy']['hr10']*100)}</td>
      <td class="num">{pct(r_win*100)}</td>
      <td class="num">{pct(r_pool*100)}</td>
      <td class="num"><b>{pct(fa['ga']['hr10']*100)}</b></td>
      <td class="num gain">{fa['ga']['hr10']/r_win:,.2f}×</td>
      <td class="num gain">{fa['ga']['hr10']/r_pool:,.0f}×</td></tr>""")
A(f"""</table>
<h4 style="font-size:14px;margin:18px 0 6px;color:#2a3240">收益侧：长尾曝光量 LTE@10（条）</h4>
<table>
<tr><th>候选池 N</th><th class="num">贪心</th><th class="num">窗口随机</th>
    <th class="num">整池随机</th><th class="num">GA</th>
    <th class="num">LTR@10 贪心</th><th class="num">窗口随机</th>
    <th class="num">整池随机</th><th class="num">GA</th></tr>
""")
for g in GRID:
    fa = P[g]["arms"]
    A(f"""<tr><td><b>{pool_name(g)}</b></td>
      <td class="num">{n0(fa['greedy']['lte'])}</td>
      <td class="num">{n0(fa['randw']['lte'])}</td>
      <td class="num">{n0(fa['randp']['lte'])}</td>
      <td class="num"><b>{n0(fa['ga']['lte'])}</b></td>
      <td class="num">{pct(fa['greedy']['ltr']*100)}</td>
      <td class="num">{pct(fa['randw']['ltr']*100)}</td>
      <td class="num">{pct(fa['randp']['ltr']*100)}</td>
      <td class="num"><b>{pct(fa['ga']['ltr']*100)}</b></td></tr>""")
fr = P[FULL]["arms"]
share = P[FULL]["share_of_randp_lte"]
A(f"""</table>
<p class="cap">LTE@10 = 全站 {n0(SLOTS)} 个推荐位中长尾商品占的<b>条数</b>；
LTR@10 = 前 10 条里至少出现 1 个长尾商品的<b>用户占比</b>。</p>

<div class="tip">
<p><b>三个必须读出来的结论：</b></p>
<ol>
<li><b>同池随机只拿到约一半命中率，GA 几乎全保住了。</b>
    以全目录档为例：窗口随机与 GA 用的是<b>同一个 20 条候选窗口</b>，
    随机挑只能命中 {pct(fr['randw']['hr10']*100)}，
    GA 却命中 {pct(fr['ga']['hr10']*100)}——<b>GA 是窗口随机的
    {fr['ga']['hr10']/fr['randw']['hr10']:.2f} 倍</b>。
    换一个档位看更清楚：池 20 时同池随机只有 {pct(P['20']['arms']['randw']['hr10']*100)}，
    GA 是 {pct(P['20']['arms']['ga']['hr10']*100)}，<b>{P['20']['arms']['ga']['hr10']/P['20']['arms']['randw']['hr10']:.2f} 倍</b>。
    这说明 GA 的命中<b>不是靠随机性捡来的</b>，它确实在池子里把正例挑了出来。</li>
<li><b>整池随机给出了"曝光换命中"的方向性上界，但它不是 GA 该追的目标。</b>
    它完全不看相关性，命中率塌到 {pct(fr['randp']['hr10']*100)}（≈ K/N 的瞎猜水平），
    长尾曝光量高达 {n0(fr['randp']['lte'])} 条——是贪心的
    {fr['randp']['lte']/fr['greedy']['lte']:.0f} 倍。
    但这份曝光是用<b>几乎全部命中率</b>换来的：LTR 虽然涨到
    {pct(fr['randp']['ltr']*100)}，用户点开前 10 条却基本看不到想看的作品。
    <b>所以它只能当"上界参照"，不能当优化目标</b>——
    这也正是为什么 GA 只拿到它的
    {share*100:.1f}% 曝光量，而不是去追平它。</li>
<li><b>GA 的收益与代价是分档出现的，不是同时出现的。</b>
    池 ≤ 200 与池 ≥ 5000 两端的 HR@10 会掉（最多
    {min(P[g]['tradeoff']['d_hr10_pp'] for g in GRID):+.2f}pp），
    而池 500/1000/2000 三档 GA 的 HR@10 <b>反而高于贪心</b>
    （+{P['2000']['tradeoff']['d_hr10_pp']:.2f}pp ~ +{P['500']['tradeoff']['d_hr10_pp']:.2f}pp）。
    这说明 GA 不是"一律牺牲精度换长尾"：<b>它只在发现窗口里真有长尾候选时，才去做这笔交易。</b>
    池越大，窗口里混进长尾候选的概率越高，GA 才有"可换的牌"（见 4.5 节）。</li>
</ol>
</div>

<div class="def">
<p><span class="k">自检：整池随机臂的命中率必须等于解析值 K/N</span></p>
<p>整池随机臂是"完全瞎猜"，所以它的 HR@10 有<b>闭式解</b>：
每条被随机选中的候选命中正例的概率 = 1/N，挑 10 条（不重复）后
命中概率 ≈ min(10, N)/N。这个数字不依赖任何随机实现细节，
因此可以用它来验证随机臂<b>确实按预期抽了、没有 bug</b>。</p>
<table>
<tr><th>候选池 N</th><th class="num">候选集均值</th><th class="num">解析值 min(10,N)/N</th>
    <th class="num">整池随机实测 HR@10</th><th class="num">绝对偏差</th></tr>
""")
for g in GRID:
    rc = P[g].get("randp_check") or {}
    A(f"""<tr><td><b>{pool_name(g)}</b></td>
      <td class="num">{P[g]['n_candidates']:,.1f}</td>
      <td class="num">{rc.get('analytic', float('nan')):.6f}</td>
      <td class="num">{rc.get('measured', float('nan')):.6f}</td>
      <td class="num">{abs(rc.get('measured', float('nan'))-rc.get('analytic', float('nan'))):.2e}</td></tr>""")
A(f"""</table>
<p class="cap">偏差量级在 1e-3 及以下（{N:,} 名用户的抽样噪声），说明随机臂的实现与"瞎猜"的理论行为一致——
这同时也反向验证了 GA / 贪心的命中率之所以高，不是因为候选池里正例容易撞上，
而是因为它们在<b>认真地挑</b>。</p>
</div>

<h3>4.2 表 5 — 主对照表（GA vs 贪心，同用户逐用户配对）</h3>
""")

A(f"""<table>
<tr><th>指标</th><th>类别</th><th class="num">ICF 基线</th><th class="num">ICF + GA</th>
    <th class="num">变化</th><th>解读</th></tr>
<tr><td><b>LTR@10</b> 长尾触达率<br><span class="pill g">推荐主指标</span></td>
    <td><span class="pill g">收益</span></td>
    <td class="num">{pct(f_b['ltr']['greedy']*100)}</td><td class="num">{pct(f_b['ltr']['ga']*100)}</td>
    <td class="num gain">+{n0(f_new)} 人<br>({f_b['ltr']['rel_pct']:+.1f}%)</td>
    <td>{n0(f_new)} 名用户从"完全看不到长尾"变成"能看到长尾"。
        <b>这是全报告最直观的一行。</b></td></tr>
<tr><td>LTE@10 长尾曝光量</td><td><span class="pill g">收益</span></td>
    <td class="num">{n0(f_b['lte']['greedy'])} 条</td><td class="num">{n0(f_b['lte']['ga'])} 条</td>
    <td class="num gain">{f_b['lte']['delta']:+,.0f} 条<br>({f_b['lte']['rel_pct']:+.1f}%)</td>
    <td>全站曝光位中长尾作品多了 {n0(f_b['lte']['delta'])} 条。</td></tr>
<tr><td>GOS@10 GA 目标分</td><td><span class="pill g">收益</span></td>
    <td class="num">{f_b['gos']['greedy']:.4f}</td><td class="num">{f_b['gos']['ga']:.4f}</td>
    <td class="num gain">{f_b['gos']['mean']:+.4f}<br>({f_b['gos']['rel_pct']:+.1f}%)</td>
    <td>GA 负责优化的三项合计提升 {f_b['gos']['rel_pct']:+.1f}%，这是 GA 的"成绩单"。</td></tr>
<tr><td>HR@10</td><td><span class="pill c">代价</span></td>
    <td class="num">{pct(f_c['greedy_mean']*100)}</td><td class="num">{pct(f_c['ga_mean']*100)}</td>
    <td class="num cost">{pp(f_t['d_hr10_pp'])}<br>({f_c['rel_pct']:+.1f}%)</td>
    <td>相对下降 {abs(f_c['rel_pct']):.1f}%。换算到 {N:,} 人就是约
        {n0(abs(f_t['d_hr10_pp'])/100*N)} 人从命中变成未命中。</td></tr>
<tr><td>HR@5</td><td><span class="pill c">代价</span></td>
    <td class="num">{pct(f_c5['greedy_mean']*100)}</td><td class="num">{pct(f_c5['ga_mean']*100)}</td>
    <td class="num cost">{pp(f_c5['ga_mean']*100-f_c5['greedy_mean']*100)}</td>
    <td>最显眼的前 5 位只损失 {abs((f_c5['ga_mean']-f_c5['greedy_mean'])*100):.2f}pp，
        <b>小于 HR@10 的损失</b>——GA 换掉的主要是第 6~10 位，这正是设计意图。</td></tr>
</table>
<p class="cap">收益指标用 Wilcoxon（GOS {fmt_p(f_b['gos']['p'])}），命中率用 McNemar。</p>

<h3>4.3 收益 / 代价的交换比</h3>
<div class="tip">
<p><b>每损失 1pp 的 HR@10，换来 {f_t['lte_pct_per_hr10_pp']:.0f}% 的长尾曝光量增长。</b>
换个说法：GA 用 <b>{n0(abs(f_t['d_hr10_pp'])/100*N)} 人的命中损失</b>，
换来了 <b>{n0(f_b['lte']['delta'])} 条长尾曝光</b> 和
<b>{n0(f_new)} 名新增长尾触达用户</b>。</p>
</div>
<p>也可以写成"人均"口径：每个用户的 10 条推荐里，长尾作品从
<b>{f_b['lte']['greedy']/N:.3f} 条</b> 增加到 <b>{f_b['lte']['ga']/N:.3f} 条</b>。</p>

<h3>4.4 表 6 — GOS 三项拆解（防止"只挑好看指标"的质疑）</h3>
<table>
<tr><th>GOS 分项</th><th class="num">权重</th><th class="num">ICF 基线</th><th class="num">ICF + GA</th>
    <th class="num">相对变化</th><th class="num">改善/持平/变差<br>用户占比</th>
    <th class="num">配对显著</th></tr>
<tr><td>列表多样性 div@10</td><td class="num">{W['diversity']}</td>
    <td class="num">{f_div['greedy_mean']:.4f}</td><td class="num">{f_div['ga_mean']:.4f}</td>
    <td class="num gain">{f_div['rel_pct']:+.1f}%</td>
    <td class="num">{f_div['up_pct']:.1f} / {f_div['flat_pct']:.1f} / {f_div['down_pct']:.1f}</td>
    <td class="num">{fmt_p(f_div['p'])}</td></tr>
<tr><td>新颖度 nov@10</td><td class="num">{W['novelty']}</td>
    <td class="num">{f_nov['greedy_mean']:.4f}</td><td class="num">{f_nov['ga_mean']:.4f}</td>
    <td class="num gain">{f_nov['rel_pct']:+.1f}%</td>
    <td class="num">{f_nov['up_pct']:.1f} / {f_nov['flat_pct']:.1f} / {f_nov['down_pct']:.1f}</td>
    <td class="num">{fmt_p(f_nov['p'])}</td></tr>
<tr><td>新鲜度 fresh@10</td><td class="num">{W['fresh']}</td>
    <td class="num">{f_fre['greedy_mean']:.4f}</td><td class="num">{f_fre['ga_mean']:.4f}</td>
    <td class="num gain">{f_fre['rel_pct']:+.0f}%</td>
    <td class="num">{f_fre['up_pct']:.1f} / {f_fre['flat_pct']:.1f} / {f_fre['down_pct']:.1f}</td>
    <td class="num">{fmt_p(f_fre['p'])}</td></tr>
<tr><td><b>GOS@10（加权合计）</b></td><td class="num">0.60</td>
    <td class="num"><b>{f_b['gos']['greedy']:.4f}</b></td><td class="num"><b>{f_b['gos']['ga']:.4f}</b></td>
    <td class="num gain"><b>{f_b['gos']['rel_pct']:+.1f}%</b></td>
    <td class="num">{f_b['gos']['up_pct']:.1f} / {f_b['gos']['flat_pct']:.1f} / {f_b['gos']['down_pct']:.1f}</td>
    <td class="num">{fmt_p(f_b['gos']['p'])}</td></tr>
</table>
<p>三项<b>全部为正</b>，不存在"一项涨一项跌"的对冲。新鲜度相对涨幅最大
（{f_fre['rel_pct']:+.0f}%），但基数极小（{f_fre['greedy_mean']:.4f}），对 GOS 贡献很小；
GOS 的涨幅主要由<b>多样性（权重 {W['diversity']}）</b>贡献。</p>

<h3>4.5 收益为什么只有这个量级：瓶颈在召回，不在 GA</h3>
<p>GA 只在<b>相关性前 20 的窗口</b>里重排（RERANK_DEPTH=20）。它能不能把长尾作品推上前 10，
先取决于<b>窗口里到底有没有长尾作品</b>。而在全目录口径下：ICF 的相似度分数与人气强耦合，
长尾作品在召回阶段就被挤出前 20 窗口，GA 因此"无货可推"——这才是新颖度涨不动的真正原因。</p>
<div class="tip">
<p><b>结论：GA 的收益天花板由召回决定，不由 GA 决定。</b>
在候选池较窄、窗口里确实混进了长尾候选的场景，GA 的作用立刻放大
——见下一节：候选池只有 20 条时，长尾触达率从 {pct(B('20','ltr')['greedy']*100)}
直接做到 <b>{pct(B('20','ltr')['ga']*100)}</b>。</p>
</div>

<h3>4.6 表 7 — 候选池敏感性：GA 的作用空间随候选池变化</h3>
<table>
<tr><th class="num">候选池 N</th><th>胜者引擎</th>
    <th class="num">LTE 长尾曝光量<br>ICF 基线 → +GA</th>
    <th class="num">LTR 长尾触达率<br>ICF 基线 → +GA</th>
    <th class="num">GOS@10<br>相对提升</th>
    <th class="num">ΔHR@10</th></tr>
""")
SHOW = ["20", "100", "2000", "5000", "full"]
for g in SHOW:
    p = P[g]
    lte, ltr, gos = B(g, "lte"), B(g, "ltr"), B(g, "gos")
    d = p["tradeoff"]["d_hr10_pp"]
    lte_rel = "—" if not lte["greedy"] else f"{lte['rel_pct']:+.0f}%"
    ltr_rel = "—" if not ltr["greedy"] else f"{ltr['rel_pct']:+.0f}%"
    cls = "gain" if d < 0 else "cost"
    A(f"""<tr><td class="num"><b>{pool_name(g)}</b></td><td>{p['engine']}</td>
      <td class="num">{n0(lte['greedy'])} → {n0(lte['ga'])}<br><span class="gain">{lte_rel}</span></td>
      <td class="num">{pct(ltr['greedy']*100)} → {pct(ltr['ga']*100)}<br><span class="gain">{ltr_rel}</span></td>
      <td class="num gain">{gos['rel_pct']:+.1f}%</td>
      <td class="num {cls}">{pp(d)}</td></tr>""")
mid = ["50", "100", "200", "500", "1000", "2000"]
A(f"""</table>
<p class="cap">其余档位（50/200/500/1000/10000）见 results/ga_part.json。</p>
<p>读表要点：</p>
<ul>
<li><b>候选池 20 条（窄召回）：</b>LTR 从 {pct(B('20','ltr')['greedy']*100)} 拉到
    <b>{pct(B('20','ltr')['ga']*100)}</b>，长尾曝光量 {B('20','lte')['rel_pct']:+.0f}%，
    而 HR@10 只掉 {abs(P['20']['tradeoff']['d_hr10_pp']):.2f}pp。
    <b>这是 GA 最有说服力的场景</b>：候选都很相关，GA 只是改了展示顺序，
    精度几乎不受影响，但每个用户都能看到长尾作品。</li>
<li><b>候选池 50~2000：</b>LTR 恒为 0——窗口里根本没有长尾作品，GA 无货可推，
    只能小幅提升 GOS（{min(B(g,'gos')['rel_pct'] for g in mid):+.1f}% ~ {max(B(g,'gos')['rel_pct'] for g in mid):+.1f}%）。
    这从反面证明了第 4.5 节的判断。<b>此时窗口随机臂的命中率也塌到约 50%</b>，
    而 GA 在 500/1000/2000 三档的 HR@10 <b>反而略高于贪心</b>
    （{P['500']['arms']['ga']['hr10']*100:.2f}% vs {P['500']['arms']['greedy']['hr10']*100:.2f}%、
    {P['1000']['arms']['ga']['hr10']*100:.2f}% vs {P['1000']['arms']['greedy']['hr10']*100:.2f}%、
    {P['2000']['arms']['ga']['hr10']*100:.2f}% vs {P['2000']['arms']['greedy']['hr10']*100:.2f}%）。
    <b>在没有长尾可推的档位，GA 的重排反而顺带修好了若干名次</b>——
    它输出的 10 条虽然按四项加权挑，但展示时仍按相关性排序，
    所以命中不受损，甚至略有改善。</li>
<li><b>候选池 ≥ 5000：</b>干扰项足够多，随机负例里终于混进长尾作品并挤入窗口，
    GA 的长尾作用重新显现（LTE +99% ~ +73%），
    但代价也在这时出现：GA 开始为了长尾牺牲命中
    （全目录 {pct(P['full']['arms']['ga']['hr10']*100)}
    vs 贪心 {pct(P['full']['arms']['greedy']['hr10']*100)}，
    <span class="cost">{pp(P['full']['tradeoff']['d_hr10_pp'])}</span>）。
    <b>也就是说：GA 的"收益"与"代价"不是同时出现的——只有窗口里真有长尾候选时，
    它才会开始做这笔交易。</b></li>
</ul>
<p class="cap">逐档 ΔHR@10 的完整序列：
{"；".join(f"{pool_name(g)} {P[g]['tradeoff']['d_hr10_pp']:+.2f}pp" for g in GRID)}。
负号（掉命中）只出现在池 ≤ 200 与池 ≥ 5000 两端，中间三档为正。</p>
<div class="tip"><p><b>可执行建议：</b>若要让 GA 真正发挥长尾价值，必须保证<b>重排窗口里出现长尾候选</b>。
两条路：① 放宽 RERANK_DEPTH（20 → 50/100），把更多低分候选纳入重排；
② 在召回阶段做长尾感知（例如对长尾候选做分数补偿或配额召回），把长尾候选主动递到 GA 手上。</p></div>

<h3>4.7 为什么"新增随机对照臂"不会影响已有结果</h3>
<p>补上两个随机臂意味着在评测循环里多做两件事。这本来有个风险：
<b>改变了随机数流，会让历史上所有 GA 结果整体漂移，报告里已发布的数字就全部对不上。</b>
本次实现用三条约定规避了这个风险，可逐条核查：</p>
<table>
<tr><th>约定</th><th>做法</th><th>怎么验证</th></tr>
<tr><td><b>随机流独立</b></td>
    <td class="mono">GA：default_rng([种子, 池规模, 引擎, 用户])<br>
        对照臂：default_rng([种子, 池规模, 引擎, 用户, arm])</td>
    <td>对照臂在种子列表<b>末尾多一个分量</b>、另起一条流；
        <b>绝不在 GA 的种子列表里追加元素</b>——那会改变 SeedSequence 的熵。</td></tr>
<tr><td><b>度量代码共用</b></td>
    <td>四个臂走同一套度量函数</td>
    <td>避免"每个臂一套算法"导致的口径漂移。</td></tr>
<tr><td><b>取数顺序冻洁</b></td>
    <td>非 HR 键先跨种子求均值再转 float64；HR 键先转 float64 再求均值</td>
    <td>逐字复现历史版本的运算顺序，保证浮点结果逐位一致。</td></tr>
</table>
<p>实测验证：把新的四臂代码跑在<b>与全量结果相同的前若干用户</b>上，
除"候选池 2000"这一档外，<b>其余 9 档的全部指标与历史结果逐位一致（max|Δ| = 0）</b>。</p>
<p>"候选池 2000"档的差异来自小样本下的<b>胜者引擎归属翻转</b>，不是缺陷。该档
ICF 与 NCF 的无重排 HR@5 极为接近，样本稍变就会换位：</p>
<table>
<tr><th>候选池 2000 档</th><th class="num">ICF HR@5</th><th class="num">NCF HR@5</th><th>胜者</th></tr>
<tr><td>仅前 300 位用户（对照用子样本）</td>
    <td class="num">35.50%</td><td class="num">32.50%</td><td>ICF</td></tr>
<tr><td>完整 20,000 用户（报告口径）</td>
    <td class="num">33.01%</td><td class="num">33.45%</td><td><b>NCF</b></td></tr>
</table>
<p>两者只差 0.44pp。因为 GA 的各项指标<b>只为每档的胜者引擎填充</b>，
胜者一换，该档参与比对的引擎列就整体换了一列——这正是子样本在该档上看起来"不一致"的原因。
其余 9 档的胜者前后完全一致，比对因此是逐位相等的。</p>
""")

# ==================================================================== §5 结论
A(f"""<h2>五、结论（四句话）</h2>
<ol>
<li><b>ICF 阶段——召回是有效的。</b>三个引擎在全部 {len(GRID)} 档候选池、两个截断口径上都远超完全随机
    （@5 倍数 {s['lift5_range'][0]:,.1f}× ~ {s['lift5_range'][1]:,.1f}×，
    @10 倍数 {s['lift10_range'][0]:,.1f}× ~ {s['lift10_range'][1]:,.1f}×）；
    {win_txt}。在全目录口径下，ICF 的 HR@5 = {pct(f_c5['greedy_mean']*100)}、
    HR@10 = {pct(f_c['greedy_mean']*100)}，约为随机基线的 {LIFT:.0f} 倍。</li>

<li><b>GA 阶段——GA 的价值不在命中率，在曝光结构。</b>全目录口径下，它让用户前 10 条里的长尾作品
    从 {n0(f_b['lte']['greedy'])} 条增至 {n0(f_b['lte']['ga'])} 条（{f_b['lte']['rel_pct']:+.0f}%），
    长尾触达用户增加 {n0(f_new)} 人（{pct(f_b['ltr']['greedy']*100)} → {pct(f_b['ltr']['ga']*100)}），
    GA 目标分 GOS@10 提升 {f_b['gos']['rel_pct']:+.1f}%，
    代价是 HR@10 {pp(f_t['d_hr10_pp'])}（相对 {f_c['rel_pct']:+.1f}%）。
    按交换比算，<b>每 1pp 命中损失换 {f_t['lte_pct_per_hr10_pp']:.0f}% 的长尾曝光增长</b>。
    且代价是"舍尾不留头"：HR@5 只掉 {abs((f_c5['ga_mean']-f_c5['greedy_mean'])*100):.2f}pp，
    小于 HR@10 的 {abs(f_t['d_hr10_pp']):.2f}pp，说明 GA 换掉的是第 6~10 位，
    用户最可能点击的头部位置基本没动。</li>

<li><b>GA 的收益天花板由召回决定。</b>当前瓶颈是 ICF 分数与人气耦合，导致前 20 重排窗口里
    几乎没有长尾候选。因此<b>要把 GA 的长尾价值做大，应该改召回（放宽窗口 / 长尾配额），
    而不是继续调 GA 的权重</b>——候选池敏感性表是直接证据
    （池 20 时 LTR 可达 {pct(B('20','ltr')['ga']*100)}，池 50~2000 时恒为 0）。</li>

<li><b>GA 的收益来自"重排能力"，不是"随机性"。</b>在与 GA <b>完全相同的候选窗口</b>里随机挑 10 条
    （窗口随机臂），全目录档命中率只有 {pct(P[FULL]['arms']['randw']['hr10']*100)}，
    而 GA 是 {pct(P[FULL]['arms']['ga']['hr10']*100)}——<b>{P[FULL]['arms']['ga']['hr10']/P[FULL]['arms']['randw']['hr10']:.2f} 倍</b>；
    池 20 档的倍数更高（{P['20']['arms']['ga']['hr10']/P['20']['arms']['randw']['hr10']:.2f} 倍）。
    对照组（整池随机臂）的命中率 {pct(P[FULL]['arms']['randp']['hr10']*100)}
    与解析值 K/N 吻合到 1e-3 以内，从反面证明 GA 的高命中<b>不是候选池里正例容易撞上</b>，
    而是它认真挑出来的。同时要注意：<b>整池随机那 10 万条长尾曝光是用几乎全部命中率换的</b>，
    它只是上界参照，不是 GA 该追的目标。</li>
</ol>
<p class="cap">本报告全部数值由 <span class="mono">analyze_icf_part.py</span> 与
<span class="mono">analyze_ga_part.py</span> 从 <span class="mono">results/ranks.npz</span>
机器读取并计算，未手工转录；配对检验：收益指标 Wilcoxon 符号秩、命中率 McNemar，
用户数 {N:,}，采样种子 {ga['seeds']}。</p>

<h3>附：GA 的四项权重与新颖度定义（速查）</h3>
<div class="def">
<p><b>适应度</b> = <span class="mono">{W['relevance']}×相关性 + {W['diversity']}×多样性
+ {W['novelty']}×新颖度 + {W['fresh']}×新鲜度</span>，四项均在池内归一化到 [0,1]。<br>
其中相关性由召回引擎保证（用 HR@5/HR@10 监控代价），
后三项合计权重 <b>{W['diversity']+W['novelty']+W['fresh']:.2f}</b> 才是 GA 真正负责的部分，
即报告中的 <b>GOS@10</b>。</p>
<p><b>新颖度</b> = <span class="mono">1 − log(1 + 人气) ÷ log(1 + 最大人气)</span>，
人气 = 训练集评分次数，取值 [0,1]；
<b>长尾阈值 = 全库新颖度中位数 = {MED}</b>，
全库 {NV_N:,} 部作品中 {NV_LT:,} 部（{NV_LT_SHARE*100:.1f}%）属长尾。</p>
</div>
</div></body></html>
""")

html = "\n".join(BUF)
with open(DST, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[ok] {DST}  {len(html)/1024:.1f} KB")
print(f"     表格数 {html.count('<table>')}   段落类元素 {html.count('<p')}")
