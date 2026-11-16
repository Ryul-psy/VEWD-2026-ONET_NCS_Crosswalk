"""세 번째 수행준거 검증표로 최종 모델 후보 다섯 가지를 비교한다.

    python evaluate_validation3.py

검증표는 읽기만 한다. 일치·좁음·넓음·근접을 "맞는 연결"로 본다.
후보는 IN·OC·RB·IB·FN의 상위 5개를 합친 것이므로 다섯 방법의 상위 5 이내 적중률은 모두 정확하다.
가중치는 판정 전에 1로 고정했다. 다른 가중치는 7절에서 하한으로만 본다 (판정되지 않은 Task는 틀린 것으로 셈).

결과: data-ONET-NCS/validation3/validation3_evaluation.md
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

import numpy as np
from openpyxl import load_workbook

from evaluate_validation2 import GOOD, KS, RELATIONS, auc, kappa, sign_p
from linkdata import ROOT, read_onet

VALID = ROOT / "data-ONET-NCS" / "validation3"
METHODS = ["IN", "OC", "RB", "IB", "FN"]
STRICT = {"일치", "좁음", "넓음"}


def read_sheet(name: str):
    rows = load_workbook(VALID / name, read_only=True)["검증"].iter_rows(values_only=True)
    header = list(next(rows))
    col = {c: header.index(c) for c in ("번호", "후보", "관계유형", "더 맞는 Task ID", "대분류")}
    label, major, better = {}, {}, {}
    for r in rows:
        if r[col["번호"]] is None:
            continue
        n = int(r[col["번호"]])
        value = r[col["관계유형"]]
        label[(n, int(r[col["후보"]]))] = value.strip() if isinstance(value, str) else ""
        major[n] = r[col["대분류"]]
        if r[col["더 맞는 Task ID"]] not in (None, ""):
            better[n] = str(r[col["더 맞는 Task ID"]]).strip()
    return label, major, better


def main() -> int:
    label, major, better = read_sheet("validation3.xlsx")
    unknown = Counter(v for v in label.values() if v and v not in RELATIONS)
    if unknown:
        print(f"목록에 없는 관계유형이 있습니다: {dict(unknown)}")
        return 1
    with (VALID / "validation3_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    by_item = defaultdict(list)
    for r in key:
        r["번호"], r["후보"] = int(r["번호"]), int(r["후보"])
        for m in METHODS:
            r[f"{m}_순위"], r[f"{m}_점수"] = int(r[f"{m}_순위"]), float(r[f"{m}_점수"])
        by_item[r["번호"]].append(r)

    def prepare(lab):
        return {n: [dict(r, 관계유형=lab[(n, r["후보"])]) for r in rs] for n, rs in by_item.items()
                if all(lab.get((n, r["후보"])) for r in rs)}

    def hit(rs, good, m, k):
        return any(r[f"{m}_순위"] <= k and r["관계유형"] in good for r in rs)

    def table(items, good):
        rows = ["| 방법 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- |" + " --- |" * len(KS)]
        for m in METHODS:
            hits = [sum(hit(rs, good, m, k) for rs in items.values()) for k in KS]
            rows.append(f"| {m} | " + " | ".join(f"{h / len(items):.1%} ({h})" for h in hits) + " |")
        return rows + [""]

    def signs(items, k):
        rows = ["| 비교 | 앞 방법만 맞힘 | 뒤 방법만 맞힘 | p |", "| --- | --- | --- | --- |"]
        for i, a in enumerate(METHODS):
            for b in METHODS[i + 1:]:
                win = sum(hit(rs, GOOD, a, k) and not hit(rs, GOOD, b, k) for rs in items.values())
                lose = sum(hit(rs, GOOD, b, k) and not hit(rs, GOOD, a, k) for rs in items.values())
                rows.append(f"| {a} 대 {b} | {win} | {lose} | {sign_p(win, lose):.3f} |")
        return rows + [""]

    done = prepare(label)
    if not done:
        print("후보가 모두 판정된 수행준거가 아직 없습니다.")
        return 1
    matched = {n: rs for n, rs in done.items() if any(r["관계유형"] in GOOD for r in rs)}
    strict = {n: rs for n, rs in done.items() if any(r["관계유형"] in STRICT for r in rs)}
    unmatched = [n for n in done if n not in matched]
    judged = sum(len(rs) for rs in done.values())
    counts = Counter(r["관계유형"] for rs in done.values() for r in rs)
    tasks = {o["Task ID"] for o in read_onet()}
    wrote = [n for n in unmatched if better.get(n) in tasks]
    none = [n for n in unmatched if better.get(n) == "없음"]

    out = ["# 수행준거 → Task 세 번째 검증 결과", "",
           f"- 판정된 쌍: {judged:,}개 / {len(key):,}개, 후보가 모두 판정된 수행준거: {len(done)}개 / {len(by_item)}개",
           f"- 맞는 연결(일치·좁음·넓음·근접)이 있는 수행준거: {len(matched)}개 ({len(matched) / len(done):.1%})",
           f"- `근접`을 빼고 일치·좁음·넓음이 있는 수행준거: {len(strict)}개 ({len(strict) / len(done):.1%})",
           f"- 맞는 연결이 없는 수행준거: {len(unmatched)}개. 그중 `더 맞는 Task ID`에 Task를 적은 것 {len(wrote)}개, "
           f"`없음` {len(none)}개, 그 밖 {len(unmatched) - len(wrote) - len(none)}개", "",
           "## 관계유형 분포", "", "| 관계유형 | 쌍 수 | 비율 |", "| --- | --- | --- |"]
    out += [f"| {name} | {counts[name]:,} | {counts[name] / judged:.1%} |" for name in RELATIONS] + [""]
    out += [f"## 1. 상위 k 적중률 (맞는 연결이 있는 수행준거 {len(matched)}개)", ""] + table(matched, GOOD)
    out += [f"## 2. 전체 수행준거 {len(done)}개 기준", "", "맞는 Task가 없는 수행준거까지 넣은 값이다.", ""] + table(done, GOOD)
    out += [f"## 3. `근접`을 빼고 본 상위 k 적중률 (수행준거 {len(strict)}개)", ""] + table(strict, STRICT)
    out += ["## 4. 방법끼리 비교: 한쪽만 맞힌 수행준거 수 (양측 부호검정)", "", "### 상위 1", ""] + signs(matched, 1)
    out += ["### 상위 5", ""] + signs(matched, 5)

    out += ["## 5. 방법별 1위 후보의 관계유형 (전체 수행준거)", "",
            "| 방법 | " + " | ".join(RELATIONS) + " |", "| --- |" + " --- |" * 6]
    for m in METHODS:
        first = Counter(r["관계유형"] for rs in done.values() for r in rs if r[f"{m}_순위"] == 1)
        out.append(f"| {m} | " + " | ".join(str(first[name]) for name in RELATIONS) + " |")
    good_rows = [r for rs in done.values() for r in rs if r["관계유형"] in GOOD]
    out += ["", "## 6. 맞는 연결이 어느 방법의 상위 5개에서 나왔나", "",
            "| 방법 | 상위 5개에 든 맞는 쌍 | 이 방법에서만 나온 맞는 쌍 |", "| --- | --- | --- |"]
    for m in METHODS:
        own = [r for r in good_rows if r[f"{m}_순위"] <= 5]
        only = [r for r in own if not any(r[f"{o}_순위"] <= 5 for o in METHODS if o != m)]
        out.append(f"| {m} | {len(own)} / {len(good_rows)} | {len(only)} |")
    out.append("")

    # 7. 다른 가중치: IN 상위 50개 전체를 다시 정렬. 판정되지 않은 Task는 틀린 것으로 센다.
    z = np.load(VALID / "work" / "short.npz")
    onet = read_onet()
    task_id = np.array([o["Task ID"] for o in onet])[z["short"]]

    def std(x):
        return (x - x.mean(1, keepdims=True)) / np.where(x.std(1, keepdims=True) == 0, 1, x.std(1, keepdims=True))

    zin, zocc, zrb = std(z["IN"]), std(z["직업"]), std(z["RB"])
    rated = {n: {r["Task ID"]: r["관계유형"] in GOOD for r in rs} for n, rs in done.items()}
    out += ["## 7. 가중치를 바꿨을 때 (하한)", "",
            "점수 = z(IN) + a·z(직업) + b·z(재순위). 판정되지 않은 Task는 틀린 것으로 세므로 고정 가중치(1)에서 멀수록 불리하다.",
            "가중치를 이 표에서 고르면 시험 자료에 맞추는 것이므로 참고용이다.", "",
            "| a (직업) | b (재순위) | 상위 1 | 상위 5 | 상위 5개 중 판정된 비율 |", "| --- | --- | --- | --- | --- |"]
    for a, b in [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0), (0, 1), (0.5, 0.5), (1, 0.5), (0.5, 1), (1, 1), (1.5, 1), (1, 1.5), (2, 1), (2, 2)]:
        order = np.argsort(-(zin + a * zocc + b * zrb), axis=1, kind="stable")[:, :5]
        top1 = top5 = seen = 0
        for n in matched:
            ids = task_id[n - 1, order[n - 1]]
            top1 += rated[n].get(ids[0], False)
            top5 += any(rated[n].get(t, False) for t in ids)
            seen += sum(t in rated[n] for t in ids)
        out.append(f"| {a} | {b} | {top1 / len(matched):.1%} ({top1}) | {top5 / len(matched):.1%} ({top5}) | "
                   f"{seen / (5 * len(matched)):.0%} |")
    out.append("")

    out += ["## 8. 1위 점수로 `1위가 맞는 연결인가`를 가려낼 수 있나 (전체 수행준거, AUC)", "",
            "| 방법 | AUC | 점수 상위 25%만 남겼을 때 1위 정확도 | 상위 50% |", "| --- | --- | --- | --- |"]
    for m in METHODS:
        top = sorted((next(r for r in rs if r[f"{m}_순위"] == 1) for rs in done.values()), key=lambda r: -r[f"{m}_점수"])
        ok = [r["관계유형"] in GOOD for r in top]
        value = auc([r[f"{m}_점수"] for r in top if r["관계유형"] in GOOD], [r[f"{m}_점수"] for r in top if r["관계유형"] not in GOOD])
        q, h = len(top) // 4, len(top) // 2
        out.append(f"| {m} | {value:.2f} | {sum(ok[:q]) / q:.1%} | {sum(ok[:h]) / h:.1%} |")
    out.append("")

    by_major = defaultdict(list)
    for n, rs in done.items():
        by_major[major[n]].append((n in matched, hit(rs, GOOD, "IN", 1), hit(rs, GOOD, "FN", 1), hit(rs, GOOD, "FN", 5)))
    out += ["## 9. 대분류별", "", "| 대분류 | 수행준거 | 맞는 연결 있음 | IN 1위 적중 | FN 1위 적중 | FN 상위 5 적중 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for name, v in sorted(by_major.items(), key=lambda x: -sum(a for a, *_ in x[1])):
        out.append(f"| {name} | {len(v)} | " + " | ".join(str(sum(t[i] for t in v)) for i in range(4)) + " |")
    out.append("")

    second = VALID / "validation3_검증자2.xlsx"
    if second.exists():
        label2, _, better2 = read_sheet(second.name)
        both = [(label[k], v) for k, v in label2.items() if v and label.get(k)]
        if both:
            binary = [(a in GOOD, b in GOOD) for a, b in both]
            items2 = prepare(label2)
            shared = sorted(set(items2) & set(done))
            has1 = {n for n in shared if n in matched}
            has2 = {n for n in shared if any(r["관계유형"] in GOOD for r in items2[n])}
            out += ["## 10. 두 검증자 일치도", "",
                    f"- 두 사람이 모두 판정한 쌍: {len(both):,}개 (수행준거 {len(shared)}개)",
                    f"- 여섯 유형이 같은 쌍: {sum(a == b for a, b in both) / len(both):.1%} (카파 {kappa(both):.2f})",
                    f"- 맞는 연결 여부가 같은 쌍: {sum(a == b for a, b in binary) / len(binary):.1%} (카파 {kappa(binary):.2f})",
                    f"- 맞는 쌍 수: 검증자 1 {sum(a for a, _ in binary)}개, 검증자 2 {sum(b for _, b in binary)}개, "
                    f"둘 다 {sum(a and b for a, b in binary)}개",
                    f"- 맞는 연결이 있는 수행준거: 검증자 1 {len(has1)}개, 검증자 2 {len(has2)}개, 둘 다 {len(has1 & has2)}개 / {len(shared)}개",
                    "", "행 = 검증자 1, 열 = 검증자 2", "", "| | " + " | ".join(RELATIONS) + " |", "| --- |" + " --- |" * 6]
            cross = Counter(both)
            out += [f"| {a} | " + " | ".join(str(cross[(a, b)]) for b in RELATIONS) + " |" for a in RELATIONS]
            out += ["", "### 검증자별 적중 수 (같은 수행준거)", "",
                    "| 방법 | 검증자 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- | --- |" + " --- |" * len(KS)]
            for who, items, has in (("1", done, has1), ("2", items2, has2)):
                for m in METHODS:
                    out.append(f"| {m} | {who} (수행준거 {len(has)}개) | "
                               + " | ".join(str(sum(hit(items[n], GOOD, m, k) for n in has)) for k in KS) + " |")
            out.append("")

    text = "\n".join(out)
    (VALID / "validation3_evaluation.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
