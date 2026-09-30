"""Screenshot pipeline check. Real frame data is absent, so real-mode recognition is not evaluated.

Synthetic mode: fixture colour-pattern frames verify decoding, encoding, gallery search, candidate
aggregation and the uncalibrated / out-of-gallery behaviour. These numbers are NOT screenshot
recognition accuracy and must never be quoted as such.
"""

from __future__ import annotations

import numpy as np

from evaluation.common import Run
from pipelines.common import read_jsonl


def run(settings, tag=None, **_) -> dict:
    from PIL import Image

    from app.services.serving import load_state

    run = Run(settings, "multimodal", {"tag": tag})
    st = load_state(settings)
    metrics: dict = {"capability": st.cap("screenshot").public()}
    qpath = settings.synthetic_dir / "evaluation" / "image_queries.jsonl" if settings.mode == "synthetic" else \
        settings.data_root / "evaluation" / "image_queries.jsonl"
    if st.images is None or not qpath.exists():
        run.status = "not_run"
        report = (f"# Screenshot identification - {settings.mode} data\n\nStatus: not_run (limited_by_data). "
                  f"{st.cap('screenshot').detail}\n")
        return run.finish(metrics, report, {})
    hits1, hits5, ood_best, in_best = [], [], [], []
    for q in read_jsonl(qpath):
        with Image.open(settings.data_root.parent / q["path"]) as img:
            res = st.images.identify(img.convert("RGB"))
        cands = [c["item_key"] for c in res["candidates"]]
        best = res["candidates"][0]["similarity"] if res["candidates"] else None
        if q["in_gallery"]:
            hits1.append(float(cands[:1] == [q["target_item_key"]]))
            hits5.append(float(q["target_item_key"] in cands[:5]))
            in_best.append(best)
        else:
            ood_best.append(best)
        run.case({"query_id": q["query_id"], "in_gallery": q["in_gallery"], "target": q.get("target_item_key"),
                  "top": cands[:5], "best_similarity": best, "calibrated": res["calibrated"]})
    metrics.update({"label_source": "synthetic_fixture", "in_gallery_queries": len(hits1),
                    "fixture_hits@1": float(np.mean(hits1)) if hits1 else None,
                    "fixture_hits@5": float(np.mean(hits5)) if hits5 else None,
                    "in_gallery_best_similarity_mean": float(np.mean(in_best)) if in_best else None,
                    "out_of_gallery_best_similarity": ood_best,
                    "threshold_calibrated": st.images.accept_threshold is not None})
    report = "\n".join([
        f"# Screenshot pipeline check - {settings.mode} data", "",
        "> Fixture frames are generated colour patterns, not anime frames. This validates the upload -> encode -> "
        "gallery search -> candidate path only. It is not a measurement of screenshot recognition accuracy.", "",
        f"- fixture Hits@1 {metrics['fixture_hits@1']}, Hits@5 {metrics['fixture_hits@5']} "
        f"({len(hits1)} in-gallery fixture queries)",
        f"- out-of-gallery query best similarity {ood_best}; no acceptance threshold is calibrated, so every result "
        "is presented as an unverified candidate.", ""])
    return run.finish(metrics, report, {"queries": len(hits1) + len(ood_best)})
