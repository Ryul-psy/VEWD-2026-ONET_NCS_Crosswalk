"""사람이 채운 검증표로 모델 3가지를 비교한다 (METHOD.md 7절).

    python evaluate_models.py            # 1번 묶음
    python evaluate_models.py --batch 2

후보가 모두 판정된 수행준거만 쓴다. 일치·좁음·넓음·근접을 "맞는 연결"로 본다.
상위 k 적중률 = 맞는 연결이 하나라도 있는 수행준거 가운데, 모델의 상위 k개 안에
맞는 연결이 들어 있던 비율. 후보를 모델별 상위 5개로 모았으므로 k는 5까지만 정확하다.

결과: data-ONET-NCS/batch1/validation/batch1_evaluation.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict

from openpyxl import load_workbook

from find_candidates import ALL_MODELS as MODELS
from linkdata import OUT

GOOD = {"일치", "좁음", "넓음", "근접"}
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
KS = [1, 2, 3, 5]
VALID = OUT / "validation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    sheet = load_workbook(VALID / f"batch{args.batch}_validation.xlsx", read_only=True)["검증"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    col = {name: header.index(name) for name in ("번호", "후보", "관계유형")}
    label = {}
    for row in rows:
        if row[col["번호"]] is None:
            continue
        value = (row[col["관계유형"]] or "").strip() if isinstance(row[col["관계유형"]], str) else ""
        label[(int(row[col["번호"]]), int(row[col["후보"]]))] = value
    unknown = Counter(v for v in label.values() if v and v not in RELATIONS)
    if unknown:
        print(f"목록에 없는 관계유형이 있습니다: {dict(unknown)}")
        return 1

    with (VALID / f"batch{args.batch}_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    by_item = defaultdict(list)
    for r in key:
        r["관계유형"] = label.get((int(r["번호"]), int(r["후보"])), "")
        by_item[int(r["번호"])].append(r)

    done = {n: rs for n, rs in by_item.items() if all(r["관계유형"] for r in rs)}
    matched = {n: rs for n, rs in done.items() if any(r["관계유형"] in GOOD for r in rs)}
    exact = {n: rs for n, rs in done.items() if any(r["관계유형"] == "일치" for r in rs)}
    judged = sum(1 for rs in by_item.values() for r in rs if r["관계유형"])
    if not done:
        print(f"후보가 모두 판정된 수행준거가 아직 없습니다. (판정된 쌍 {judged:,}개 / {len(key):,}개)")
        return 1

    def accuracy(items, good, model, k):
        hits = sum(any(r.get(f"{model}_순위") and int(r[f"{model}_순위"]) <= k and r["관계유형"] in good
                       for r in rs) for rs in items.values())
        return hits / len(items) if items else float("nan")

    out = [f"# {args.batch}번 묶음 모델 비교", ""]
    out += [f"- 판정된 쌍: {judged:,}개 / {len(key):,}개",
            f"- 후보가 모두 판정된 수행준거: {len(done)}개 / {len(by_item)}개",
            f"- 그중 맞는 연결(일치·좁음·넓음·근접)이 하나라도 있는 수행준거: {len(matched)}개",
            f"- 그중 `일치`가 있는 수행준거: {len(exact)}개",
            f"- 대응 Task가 없는 수행준거: {len(done) - len(matched)}개", ""]
    counts = Counter(r["관계유형"] for rs in done.values() for r in rs)
    out += ["## 관계유형 분포", "", "| 관계유형 | 쌍 수 |", "| --- | --- |"]
    out += [f"| {name} | {counts[name]:,} |" for name in RELATIONS] + [""]

    for title, items, good in [("맞는 연결(일치·좁음·넓음·근접) 기준", matched, GOOD),
                               ("`일치` 기준", exact, {"일치"})]:
        out += [f"## 상위 k 적중률: {title} (수행준거 {len(items)}개)", "",
                "| 모델 | " + " | ".join(f"상위 {k}" for k in KS) + " |",
                "| --- |" + " --- |" * len(KS)]
        for m in MODELS:
            out.append(f"| {m} | " + " | ".join(f"{accuracy(items, good, m, k):.1%}" for k in KS) + " |")
        out.append("")

    out += ["## 모델 1위 후보의 관계유형", "", "| 모델 | " + " | ".join(RELATIONS) + " |",
            "| --- |" + " --- |" * len(RELATIONS)]
    for m in MODELS:
        first = Counter(r["관계유형"] for rs in done.values() for r in rs if r.get(f"{m}_순위") == "1")
        out.append(f"| {m} | " + " | ".join(str(first[name]) for name in RELATIONS) + " |")
    out.append("")

    best = max(MODELS, key=lambda m: (accuracy(matched, GOOD, m, 1), accuracy(matched, GOOD, m, 5)))
    out += [f"상위 1 적중률이 가장 높은 모델: **{best}**", ""]

    text = "\n".join(out)
    (VALID / f"batch{args.batch}_evaluation.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
