# -*- coding: utf-8 -*-
"""
make_word.py — 把 results/ 下的 HTML 报告转成 Word（.docx）。

必须用**托管 venv** 运行（html4docx 只装在那里，系统 Python 没有）：

    "D:/workspace/2026-09-09-14-27-14/.venv-html-to-docx/Scripts/python.exe" make_word.py
    ... make_word.py ICF_GA_报告        # 只转指定报告

转换后的 docx 同时写一份到项目根目录，方便直接提交或拖进 PPT。
若根目录的旧文件正被 Word 占用，会自动另存为 <名称>_new.docx——
绝不覆盖正在被读取的文件。
"""
import os
import shutil
import sys
import time

from html4docx import HtmlToDocx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")

REPORTS = ["ICF_GA_报告"]


def convert(stem):
    src = os.path.join(OUT_DIR, stem + ".html")
    dst = os.path.join(OUT_DIR, stem + ".docx")
    if not os.path.exists(src):
        print(f"[skip] {stem}：找不到 {src}")
        return None
    t = time.time()
    doc = HtmlToDocx()
    doc.parse_html_file(src, dst)
    if not os.path.exists(dst):
        doc.save(dst)
    if not os.path.exists(dst):
        print(f"[FAIL] {stem}：转换后没有产出文件")
        return None
    root_copy = os.path.join(HERE, stem + ".docx")
    note = ""
    try:
        shutil.copyfile(dst, root_copy)
    except PermissionError:
        # 目标被占用（Word / 预览面板打开着）→ 换名字，绝不覆盖正在被读的文件
        root_copy = os.path.join(HERE, stem + "_new.docx")
        try:
            shutil.copyfile(dst, root_copy)
            note = "   （原文件被占用，已另存）"
        except Exception as e:                                   # noqa: BLE001
            root_copy, note = f"(复制失败: {e})", ""
    except Exception as e:                                       # noqa: BLE001
        root_copy, note = f"(复制失败: {e})", ""
    print(f"[ok] {stem}.docx  {os.path.getsize(dst)/1024:.1f} KB  {time.time()-t:.1f}s")
    print(f"     -> {dst}")
    print(f"     -> {root_copy}{note}")
    return dst


def main():
    wanted = sys.argv[1:] or REPORTS
    ok = sum(1 for stem in wanted if convert(stem))
    print(f"[done] {ok}/{len(wanted)} 个报告转换成功")


if __name__ == "__main__":
    main()
