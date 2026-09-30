"""Matched before/after report for the enlarged, independently held-out cohort."""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, read_json, read_rows, write_json
from .metrics import paired_bootstrap

METHODS = ("A", "B", "C", "D")


def table_rows(summary, split="test"):
    return {row["method"]: row for row in summary["summaries"] if row["source"] == "mal_recommendations" and row["split"] == split}


def report(output=None):
    output = output or ROOT / "results"
    summary = read_json(output / "summary.json")
    baseline = read_json(output / "baseline/summary.json")
    before, after = table_rows(baseline), table_rows(summary)
    selection = read_json(ROOT / "models/optimization_selection.json")
    protocol = read_json(ROOT / "data/optimization_protocol.json")
    trials = read_json(output / "validation_trials.json")
    audit = read_json(ROOT / "data/collection_audit.json")
    counts = Counter((row["split"], row["status"]) for row in audit)
    old = read_json(ROOT / "history/v1/results/summary.json")
    n_old = table_rows(old)["A"]["n_targets"]
    n_test = after["A"]["n_targets"]
    metadata_audit = read_json(ROOT / "data/provenance.json")["audit"]
    per_before = list(read_rows(output / "baseline/per_query_metrics.jsonl"))
    per_after = list(read_rows(output / "per_query_metrics.jsonl"))
    improvements = {}
    for method in METHODS:
        left = sorted([row for row in per_before if row["source"] == "mal_recommendations" and row["split"] == "test" and row["method"] == method], key=lambda row: row["target_id"])
        right = sorted([row for row in per_after if row["source"] == "mal_recommendations" and row["split"] == "test" and row["method"] == method], key=lambda row: row["target_id"])
        assert [row["target_id"] for row in left] == [row["target_id"] for row in right]
        improvements[method] = dict(coverage_delta_pp=100 * (after[method]["coverage_at_5"] - before[method]["coverage_at_5"]),
                                    ndcg_delta=after[method]["ndcg_at_5"] - before[method]["ndcg_at_5"],
                                    paired_95pct_delta_ci={metric: paired_bootstrap([row[metric] for row in left], [row[metric] for row in right]) for metric in ("coverage_at_5", "ndcg_at_5")})
    write_json(output / "before_after.json", dict(test_targets=n_test, before=before, after=after, improvements=improvements,
                                                 comparison="identical fresh test targets, truths and full candidate catalog"))
    with (output / "before_after.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "test_targets", "before_coverage", "after_coverage", "coverage_delta_pp", "before_ndcg", "after_ndcg", "ndcg_delta"])
        for method in METHODS:
            writer.writerow([method, n_test, before[method]["coverage_at_5"], after[method]["coverage_at_5"], improvements[method]["coverage_delta_pp"], before[method]["ndcg_at_5"], after[method]["ndcg_at_5"], improvements[method]["ndcg_delta"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    x = np.arange(4)
    for axis, metric, title in zip(axes, ["coverage_at_5", "ndcg_at_5"], ["Coverage @5 (slide HR@5)", "NDCG @5"]):
        first = [before[m][metric] for m in METHODS]
        second = [after[m][metric] for m in METHODS]
        a = axis.bar(x - 0.19, first, 0.36, label="Original recipe", color="#9cb2cb")
        b = axis.bar(x + 0.19, second, 0.36, label="Validation-selected", color="#226d96")
        for bars, values in ((a, first), (b, second)):
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=3)
        axis.set_xticks(x, METHODS)
        axis.set_ylim(0, max(0.05, max(first + second) * 1.32))
        axis.set_title(title, fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)
    fig.suptitle(f"Algorithm 1 | same fresh test queries n={n_test} | full 14,202-title catalog")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    svg = output / "comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    passing = [m for m, decision in summary["replacement_decisions"].items() if decision["passes"]]
    lines = ["# 算法①：扩大样本后的优化实验（V2）", "",
             f"生成时间（UTC）：{summary['created_at']}。数据目录 14,202 部作品，K=5；每个目标均从其余 14,201 部中检索。", "",
             "## 1. 本轮结果", "",
             f"**独立测试规模从 V1 的 {n_old} 部扩大到本轮 {n_test} 部（{n_test/n_old:.1f} 倍）**；另有 {selection['validation_n']} 部新验证目标用于选型。原目标全部单列为 historical，不进入本轮模型选择或最终测试结论。人工标注仍待补齐，本轮按用户授权继续使用真实 MAL 推荐。", "",
             f"这里的 {n_test} 个测试目标是 {n_test} 部不同动漫，每种算法为每部动漫各生成一次 Top-5，然后按目标平均指标；不是把同一部动漫重复测试 {n_test} 次。", "",
             "下表的前后比较使用**同一批新测试目标、同一份真值和同一候选库**。原配方 B 也重新训练并排除了新增目标，避免与优化版形成训练泄漏差异。不能将旧报告的 38 个目标分数直接与本表相减后称为优化收益。", "",
             "| 算法 | 原配方覆盖率@5 | 优化后覆盖率@5 | 变化（百分点） | 原配方 NDCG@5 | 优化后 NDCG@5 |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        lines.append(f"| {m} | {before[m]['coverage_at_5']:.2%} | {after[m]['coverage_at_5']:.2%} | {improvements[m]['coverage_delta_pp']:+.2f} | {before[m]['ndcg_at_5']:.4f} | {after[m]['ndcg_at_5']:.4f} |")
    lines += ["", "A 的输入、权重和公式完全固定，因此前后结果相同。B/C/D 的配置只由新验证集决定，测试结果没有反过来用于调参。", "",
              "本轮 B 的两项测试指标均有提高；C 的验证集提升没有在新测试集复现；D 的 NDCG 略升而覆盖率略降。各方案前后差值的 95% 区间均跨零，尚不能确认稳定提升，区间详见第 4 节。依据既定门槛，仍建议保留 A 作为系统基线。", "",
              "![同一测试集上的优化前后对比](comparison.png)", "",
              (f"按图片的正式替换规则，本轮通过的候选为 **{', '.join(passing)}**；这是 MAL 自动评测范围内的判断，人工主要验收仍未完成。" if passing else "按图片的正式替换规则，**本轮仍没有候选同时达到两项门槛**。某个方案有所改善或数值最好，不自动等于可以正式替换。"), "",
              "| 优化方案相对 A | 覆盖率变化（百分点） | NDCG 变化 | 覆盖率 ≥+2pp 且 NDCG 不降 |", "|---|---:|---:|---|"]
    for m, decision in summary["replacement_decisions"].items():
        lines.append(f"| {m} | {decision['coverage_delta_pp']:+.2f} | {decision['ndcg_delta']:+.4f} | {'通过' if decision['passes'] else '未通过'} |")
    lines += ["", "## 2. 样本如何扩大，以及独立性如何保证", "",
              "预先固定 600 个新目标：会员数四分位各随机抽取 75 个（共 300），再从会员数前 2,000 的作品中随机抽取 300 个作为标签覆盖较好的补充组。所有新目标排除 V1 的 120 个目标；每组按固定哈希分成 1/3 验证、2/3 测试。抽样使用热度只为构建评测集，算法打分不读取热度。", "",
              "固定计划为 200 个新验证目标、400 个新测试目标。实际无足够推荐或页面不可用的目标保留审计记录，不根据算法得分删样本。", "",
              "| 分组 | 计划目标 | 有效真值 | 推荐不足 3 条 | 请求或页面失败 |", "|---|---:|---:|---:|---:|"]
    for split, count in [("historical", 120), ("validation", 200), ("test", 400)]:
        lines.append(f"| {split} | {count} | {counts[(split,'eligible')]} | {counts[(split,'insufficient_labels')]} | {counts[(split,'request_failed')]} |")
    lines += ["", "分层测试结果（应与总表一起阅读，热门补充组不代表整个目录）：", "",
              "| 抽样组 | 测试目标数 | A 覆盖率 | B 覆盖率 | C 覆盖率 | D 覆盖率 |", "|---|---:|---:|---:|---:|---:|"]
    strata = {}
    for row in summary["strata"]:
        if row["source"] == "mal_recommendations" and row["split"] == "test":
            strata.setdefault(row["sampling_group"], {})[row["method"]] = row
    for group, values in sorted(strata.items()):
        lines.append(f"| {group} | {values['A']['n_targets']} | " + " | ".join(f"{values[m]['coverage_at_5']:.2%}" for m in METHODS) + " |")
    lines += ["", f"优化协议在 `{protocol['frozen_at']}` 冻结；模型选择在 `{selection['frozen_at']}` 冻结。协议、验证逐目标结果、候选得分和选择模型均有 SHA-256。最终评测脚本要求所有 720 个目标已被审计，并验证选择后协议、划分和验证真值未发生变化。", "",
              "逻辑回归的原配方和优化配方都排除全部 720 个目标作为训练对端点，只读取原训练评分矩阵；不读取留出的评分。共同高分仅是弱训练标签，不是相似性真值。", "",
              "## 3. 优化仍在三张图片和仓库算法①范围内", "",
              "共同输入仍只有 genre、theme、demographic、studio、episodes。前两维集合去重后做 Jaccard；受众和工作室是单值精确匹配；集数使用 `1/(1+abs(e1-e2)/12)`，未知沿用 0.5。A 的固定权重仍为 0.35/0.30/0.15/0.10/0.10。没有加入简介、标题含义、用户画像、MAL 推荐边或系列关系作为预测特征，没有过滤候选或修改 Top-5。", "",
              f"**B 选中 `{selection['B']}`**：仍为同五维特征的逻辑回归。比较了原均匀随机负例、按正例作品出现频率抽取的随机负例，以及系数是否限制为非负。按出现频率抽取仍属于随机负例，只减少热门/冷门分布差异。正例仍是评分 ≥8 的共同高分作品对；两种样本各 50,000 正例＋50,000 负例。L2 正则 C=1。", "",
              f"**C 选中 `{selection['C']}`**：仍采用固定版本 `BAAI/bge-small-en-v1.5`、384 维 CLS 向量和余弦。比较原五维属性文本、紧凑标签文本、自然句文本，各自再比较是否减去无标签全目录向量均值并重新 L2 归一化。均值校正没有增加维度，也未用推荐标签训练 BGE。没有采用验证效果更差的文本写法。", "",
              f"**D 选中 `{selection['D']}`**：仍为 `0.6×A + 0.4×semantic`。比较固定 `(cos+1)/2` 与无标签目录随机作品对的 1%/99% 余弦分位映射；两路均归一化到 [0,1]，超出分位范围的值截断。后者只校准分数尺度，不训练额外模型，不改 0.6/0.4。C 和 D 共用同一份选定语义向量。", "",
              "预先限定的验证候选及实测结果如下，选择目标为 `0.5×(覆盖率@5 + NDCG@5)`，精确并列按候选登记顺序：", "",
              "| 类别 | 候选 | 验证覆盖率@5 | 验证 NDCG@5 | 采用 |", "|---|---|---:|---:|---|"]
    for trial in trials:
        selected = trial["family"] == "A" or selection.get(trial["family"]) == trial["variant"]
        lines.append(f"| {trial['family']} | {trial['variant']} | {trial['coverage_at_5']:.2%} | {trial['ndcg_at_5']:.4f} | {'是' if selected else ''} |")
    lines += ["", "## 4. 指标口径、统计区间与实际限制", "",
              "图片的 HR@5 按其文字定义计算为 `|Top5∩真值|/|真值|`，即覆盖率/Recall@5；同时另报标准 Hit Rate@5（至少命中一个）。NDCG@5 使用二元相关性，按每个目标宏平均。MAL 真值仍按用户推荐人数取前 5 个目录内作品，至少 3 个；没有通过扩充容易命中的真值或缩小候选池来抬高分数。", "",
              "同目标成对 bootstrap（3,000 次）的优化前后差值 95% 区间如下；区间跨零说明提升仍不稳定：", "",
              "| 方法前后变化 | 覆盖率变化区间（百分点） | NDCG 变化区间 |", "|---|---|---|"]
    for m in ("B", "C", "D"):
        intervals = improvements[m]["paired_95pct_delta_ci"]
        low, high = intervals["coverage_at_5"]
        nl, nh = intervals["ndcg_at_5"]
        lines.append(f"| {m} | [{low*100:+.2f}, {high*100:+.2f}] | [{nl:+.4f}, {nh:+.4f}] |")
    lines += ["", "分数仍可能不高：五维属性相同不代表剧情相似，原数据也有空主题、空受众和多工作室投影损失。MAL 的前 5 个推荐是不完整正例集合，其他合理推荐也会被记为未命中。扩大样本和优化计算不能消除这些限制；本报告保留所有不利结果。直接前作/续作关系只作为独立 sanity check，完整结果在 `metrics.csv`，不参与选型或替换判断。", "",
              f"目录审计：{metadata_audit['missing_or_empty_themes']:,} 部主题为空，{metadata_audit['missing_or_empty_demographic']:,} 部受众为空，{metadata_audit['missing_or_empty_studio']:,} 部工作室为空；{metadata_audit['multi_studio']:,} 部有多个工作室，按预定规则只取源数组第一个。空标签相互匹配仍记 0，没有为提高分数而补写事实。", "",
              "## 5. 验证与复现", "",
              "主命令：`python verify_delivery.py` → `python run_benchmark.py` → `python run_checks.py`。评测重放使用随包数据和向量，可离线执行。采样协议、验证试验、原配方结果、优化结果、逐目标预测、原网页/API 证据、完整测试输出都随文件夹交付。", "",
              "全部核心和实物数据测试的实际输出在 `tests.txt`。新增测试覆盖五维输入边界、均值校正、非负逻辑回归、目标分组隔离、两套训练样本隔离、验证选型证据、相同测试集前后对比以及 A 结果不变。浏览结果页可切换原配方/优化配方、目标作品和数据划分。", "",
              "人工标注表已扩大到 720 个预定目标，填写方式见上级 README。继续遵守人工主要验收、MAL 自动评测、系列关系 sanity check 的不同角色。未改动应用后端、前端或数据库，未进行系统联调。", "",
              "历史材料：[V1 原报告](../history/v1/results/REPORT.md)。完整旧实现由提交 `d0824db` 保留。", ""]
    # REPORT lives in results/, so history is one level above results.
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    payload = dict(before=before, after=after, selection=selection, passing=passing,
                   predictions={"baseline": list(read_rows(output / "baseline/predictions.jsonl")), "optimized": list(read_rows(output / "predictions.jsonl"))},
                   labels=[row for row in read_rows(output / "ground_truth_used.jsonl") if row["source"] == "mal_recommendations"])
    render_viewer(output, payload, n_test)
    print(f"V2 report written: fresh test n={n_test}, passing replacement candidates={passing}", flush=True)


def render_viewer(output, payload, n_test):
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>算法①优化实验 V2</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f3f6fa;color:#182b46;margin:0}main{max-width:1320px;margin:auto;padding:30px}h1{color:#184a79}section,.card{background:white;border-radius:12px;padding:24px;margin:18px 0}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #dce3ec;text-align:left}select{font:inherit;padding:8px;margin:5px;max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{border-top:4px solid #236c97;margin-top:0;padding:16px}.hit{color:#197449;font-weight:600}li{margin-bottom:16px}small{color:#627187}img{max-width:100%}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}table{font-size:13px}}@media(max-width:550px){.grid{grid-template-columns:1fr}}</style>
<main><h1>算法① · 扩大样本与五维相似度优化</h1><p>TEST_N 个新测试目标 · 14,202 部候选作品 · 原配方与优化配方在同一目标上对比 · 人工验收待补</p>
<section><h2>新测试集结果</h2><div id="metrics"></div><p>覆盖率为图片定义的 HR@5（Recall@5）；替换门槛仍是相对 A 至少 +2 个百分点且 NDCG 不下降。</p><p id="decision"></p><img src="comparison.png" alt="同一测试集上的优化前后对比"></section>
<section><h2>逐目标对比</h2><label>划分 <select id="split"><option value="test">新测试集</option><option value="validation">新验证集</option><option value="historical">历史对照</option></select></label><label>方案 <select id="variant"><option value="optimized">优化配方</option><option value="baseline">原配方</option></select></label><br><select id="target" aria-label="目标动漫"></select><p id="truth"></p><small>绿色为命中 MAL 真值。未命中仍可能是合理推荐；各模型分数仅供内部排序。</small></section><div class="grid" id="cards"></div>
<section><a href="REPORT.md">完整报告</a> · <a href="before_after.csv">同集前后指标</a> · <a href="../data/human_annotations_template.csv">人工标注表</a></section></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),names={A:'A · 五维加权',B:'B · 逻辑回归',C:'C · BGE',D:'D · 标签＋语义'};
const table=document.createElement('table'),head=document.createElement('tr');for(const s of ['方法','原覆盖率','优化覆盖率','原 NDCG','优化 NDCG']){let e=document.createElement('th');e.textContent=s;head.append(e)}table.append(head);for(const m of ['A','B','C','D']){const b=data.before[m],a=data.after[m],tr=document.createElement('tr');for(const s of [names[m],(100*b.coverage_at_5).toFixed(2)+'%',(100*a.coverage_at_5).toFixed(2)+'%',b.ndcg_at_5.toFixed(4),a.ndcg_at_5.toFixed(4)]){let e=document.createElement('td');e.textContent=s;tr.append(e)}table.append(tr)}document.getElementById('metrics').append(table);document.getElementById('decision').textContent=data.passing.length?'MAL 替换门槛通过：'+data.passing.join('、')+'；人工主验收仍待完成。':'本轮暂无方案同时通过两项替换门槛；人工主验收仍待完成。';
const target=document.getElementById('target'),split=document.getElementById('split'),variant=document.getElementById('variant');function targets(){let old=target.value;target.replaceChildren();for(const label of data.labels.filter(x=>x.split===split.value)){let row=data.predictions.optimized.find(x=>x.target_id===label.target_id),o=document.createElement('option');o.value=label.target_id;o.textContent=label.target_id+' · '+row.target_title;target.append(o)}if([...target.options].some(o=>o.value===old))target.value=old;draw()}
function draw(){let aid=Number(target.value),truth=data.labels.find(x=>x.target_id===aid),hits=new Set(truth?truth.relevant_ids:[]);document.getElementById('truth').textContent=truth?'MAL 真值 ID：'+truth.relevant_ids.join('、'):'';let cards=document.getElementById('cards');cards.replaceChildren();for(const row of data.predictions[variant.value].filter(x=>x.target_id===aid)){let card=document.createElement('div');card.className='card';let h=document.createElement('h3');h.textContent=names[row.method];card.append(h);let ol=document.createElement('ol');for(const item of row.items){let li=document.createElement('li');li.className=hits.has(item.anime_id)?'hit':'';let name=document.createElement('div');name.textContent=item.title+' ['+item.anime_id+']';let detail=document.createElement('small');detail.textContent='score '+item.similarity.toFixed(5)+' · '+Object.entries(item.dimensions).map(([key,value])=>key+' '+value.toFixed(2)).join(' / ');li.append(name,detail);ol.append(li)}card.append(ol);cards.append(card)}}split.addEventListener('change',targets);variant.addEventListener('change',draw);target.addEventListener('change',draw);targets();
</script></html>'''
    (output / "index.html").write_text(page.replace("TEST_N", str(n_test)).replace("PAYLOAD", data), encoding="utf-8", newline="\n")
