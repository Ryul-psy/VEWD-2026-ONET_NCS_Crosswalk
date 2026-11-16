"""세분류 → O*NET 직업 검증표로 후보 방법 네 가지를 비교하고, 두 검증자의 일치도를 본다.

    python evaluate_occ.py          # 1번 직업 검증표 (occ1)
    python evaluate_occ.py occ2     # 2번 직업 검증표 (나머지 990개 세분류)

검증표는 읽기만 한다. 일치·좁음·넓음·근접을 "맞는 연결"로 본다.
상위 k 적중률 = 맞는 연결이 하나라도 있는 세분류 가운데, 그 방법의 상위 k개 안에 맞는 연결이 있던 비율.
occ1은 후보를 방법별 상위 5개로 모았으므로 k는 5까지 정확하다.
occ2는 투표 상위 5개 + 이름·평균 상위 3개만 모았으므로, 투표는 상위 5까지, 이름·평균은 상위 3까지 정확하고
그 밖(이름·평균의 상위 5, 목록 전체)은 판정되지 않은 직업을 틀린 것으로 센 하한이다.

결과: data-ONET-NCS/batch1/occupation/occ1_evaluation.md (또는 occ2_evaluation.md)
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from math import comb

from openpyxl import load_workbook

from linkdata import OUT

OCC = OUT / "occupation"
METHODS = ["이름", "목록", "평균", "투표"]
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
GOOD = {"일치", "좁음", "넓음", "근접"}
KS = [1, 2, 3, 5]


def read_sheet(name: str) -> tuple[dict, dict, dict]:
    """(번호, 후보) → 관계유형, 번호 → 구분, 번호 → 검증자가 적은 `더 맞는 직업`."""
    rows = load_workbook(OCC / name, read_only=True)["검증"].iter_rows(values_only=True)
    header = list(next(rows))
    col = {c: header.index(c) for c in ("번호", "구분", "후보", "관계유형", "더 맞는 직업", "대분류")}
    label, part, better, major = {}, {}, defaultdict(list), {}
    for r in rows:
        if r[col["번호"]] is None:
            continue
        n = int(r[col["번호"]])
        value = r[col["관계유형"]]
        label[(n, int(r[col["후보"]]))] = value.strip() if isinstance(value, str) else ""
        part[n], major[n] = r[col["구분"]], r[col["대분류"]]
        if r[col["더 맞는 직업"]]:
            better[n].append(str(r[col["더 맞는 직업"]]))
    return label, part, better, major


def sign_p(win: int, lose: int) -> float:
    n, k = win + lose, min(win, lose)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(left[c] * right[c] for c in left) / n ** 2
    return (agree - chance) / (1 - chance)


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "occ1"
    label, part, better, major = read_sheet(f"{prefix}_validation.xlsx")
    unknown = Counter(v for v in label.values() if v and v not in RELATIONS)
    if unknown:
        print(f"목록에 없는 관계유형이 있습니다: {dict(unknown)}")
        return 1
    with (OCC / f"{prefix}_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    by_item = defaultdict(list)
    for r in key:
        r["관계유형"] = label.get((int(r["번호"]), int(r["후보"])), "")
        by_item[int(r["번호"])].append(r)
    done = {n: rs for n, rs in by_item.items() if all(r["관계유형"] for r in rs)}
    if not done:
        print("후보가 모두 판정된 세분류가 아직 없습니다.")
        return 1
    matched = {n: rs for n, rs in done.items() if any(r["관계유형"] in GOOD for r in rs)}
    exact = {n: rs for n, rs in done.items() if any(r["관계유형"] == "일치" for r in rs)}
    many_exact = [n for n, rs in done.items() if sum(r["관계유형"] == "일치" for r in rs) > 1]

    def hit(rs, good, method, k):
        return any(r[f"{method}_순위"] and int(r[f"{method}_순위"]) <= k and r["관계유형"] in good for r in rs)

    def accuracy(items, good, method, k):
        return sum(hit(rs, good, method, k) for rs in items.values()) / len(items) if items else float("nan")

    judged = sum(bool(r["관계유형"]) for r in key)
    out = [f"# 세분류 → O*NET 직업 검증 결과 ({prefix})", "",
           f"- 판정된 쌍: {judged:,}개 / {len(key):,}개, 후보가 모두 판정된 세분류: {len(done)}개 / {len(by_item)}개",
           f"- 맞는 연결(일치·좁음·넓음·근접)이 있는 세분류: {len(matched)}개 ({len(matched) / len(done):.1%})",
           f"- `일치`가 있는 세분류: {len(exact)}개 ({len(exact) / len(done):.1%}), `일치`가 둘 이상인 세분류: {len(many_exact)}개"
           + (f" (번호 {', '.join(map(str, many_exact))})" if many_exact else ""),
           f"- 맞는 연결이 없는 세분류: {len(done) - len(matched)}개, "
           f"그중 검증자가 `더 맞는 직업`을 적은 세분류: {sum(n in better for n in done if n not in matched)}개",
           f"- 세분류 하나에 맞는 직업 수: 평균 "
           f"{sum(sum(r['관계유형'] in GOOD for r in rs) for rs in matched.values()) / len(matched):.1f}개", ""]
    if prefix != "occ1":
        out += ["후보는 투표 상위 5개 + 이름·평균 상위 3개다. 투표는 상위 5까지, 이름·평균은 상위 3까지 정확하고, "
                "이름·평균의 상위 5와 목록은 하한이다.", ""]
    counts = Counter(r["관계유형"] for rs in done.values() for r in rs)
    out += ["## 관계유형 분포", "", "| 관계유형 | 쌍 수 | 비율 |", "| --- | --- | --- |"]
    out += [f"| {name} | {counts[name]:,} | {counts[name] / judged:.1%} |" for name in RELATIONS] + [""]

    for title, items, good in [("맞는 연결(일치·좁음·넓음·근접) 기준", matched, GOOD), ("`일치` 기준", exact, {"일치"})]:
        out += [f"## 상위 k 적중률: {title} (세분류 {len(items)}개)", "",
                "| 방법 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- |" + " --- |" * len(KS)]
        for m in METHODS:
            out.append(f"| {m} | " + " | ".join(f"{accuracy(items, good, m, k):.1%}" for k in KS) + " |")
        out.append("")

    out += ["## 방법끼리 비교: 상위 1에서 한쪽만 맞힌 세분류 수 (양측 부호검정)", "",
            "| 비교 | 앞 방법만 맞힘 | 뒤 방법만 맞힘 | p |", "| --- | --- | --- | --- |"]
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            win = sum(hit(rs, GOOD, a, 1) and not hit(rs, GOOD, b, 1) for rs in matched.values())
            lose = sum(hit(rs, GOOD, b, 1) and not hit(rs, GOOD, a, 1) for rs in matched.values())
            out.append(f"| {a} 대 {b} | {win} | {lose} | {sign_p(win, lose):.2f} |")
    out.append("")

    out += ["## 방법별 1위 후보의 관계유형", "", "| 방법 | " + " | ".join(RELATIONS) + " |", "| --- |" + " --- |" * 6]
    for m in METHODS:
        first = Counter(r["관계유형"] for rs in done.values() for r in rs if r[f"{m}_순위"] == "1")
        out.append(f"| {m} | " + " | ".join(str(first[name]) for name in RELATIONS) + " |")
    out.append("")

    out += ["## 맞는 연결이 어느 방법의 상위 5개에서 나왔나", "",
            "| 방법 | 상위 5개에 든 맞는 쌍 | 이 방법에서만 나온 맞는 쌍 |", "| --- | --- | --- |"]
    good_rows = [r for rs in done.values() for r in rs if r["관계유형"] in GOOD]
    for m in METHODS:
        own = [r for r in good_rows if r[f"{m}_순위"] and int(r[f"{m}_순위"]) <= 5]
        only = [r for r in own if not any(r[f"{o}_순위"] and int(r[f"{o}_순위"]) <= 5 for o in METHODS if o != m)]
        out.append(f"| {m} | {len(own)} / {len(good_rows)} | {len(only)} |")
    out.append("")

    by_major = defaultdict(list)
    for n in done:
        by_major[major[n]].append(n in matched)
    out += ["## 대분류별 맞는 연결이 있는 세분류", "", "| 대분류 | 세분류 수 | 맞는 연결 있음 |", "| --- | --- | --- |"]
    out += [f"| {name} | {len(v)} | {sum(v)} |" for name, v in sorted(by_major.items(), key=lambda x: sum(x[1]) / len(x[1]))]
    out.append("")

    # 두 검증자 일치도
    second = OCC / f"{prefix}_validation_검증자2.xlsx"
    if second.exists():
        label2, _, _, _ = read_sheet(second.name)
        both = [(label[k], v) for k, v in label2.items() if v and label.get(k)]
        if both:
            binary = [(a in GOOD, b in GOOD) for a, b in both]
            items2 = defaultdict(list)
            for (n, _), v in label2.items():
                items2[n].append(v)
            full2 = {n for n, vs in items2.items() if all(vs) and n in done}
            out += ["## 두 검증자 일치도", "",
                    f"- 두 사람이 모두 판정한 쌍: {len(both):,}개 (세분류 {len(full2)}개)",
                    f"- 여섯 유형이 같은 쌍: {sum(a == b for a, b in both) / len(both):.1%} (카파 {kappa(both):.2f})",
                    f"- 맞는 연결 여부가 같은 쌍: {sum(a == b for a, b in binary) / len(binary):.1%} (카파 {kappa(binary):.2f})",
                    f"- 맞는 연결이 있는 세분류: 검증자 1 {sum(n in matched for n in full2)}개, "
                    f"검증자 2 {sum(any(v in GOOD for v in items2[n]) for n in full2)}개 / {len(full2)}개", "",
                    "행 = 검증자 1, 열 = 검증자 2", "", "| | " + " | ".join(RELATIONS) + " |", "| --- |" + " --- |" * 6]
            table = Counter(both)
            out += [f"| {a} | " + " | ".join(str(table[(a, b)]) for b in RELATIONS) + " |" for a in RELATIONS]
            out += ["", "### 검증자 2의 판정으로 본 상위 k 적중률 (같은 세분류)", "",
                    "| 방법 | 검증자 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- | --- |" + " --- |" * len(KS)]
            for who, lab in (("1", label), ("2", label2)):
                items = {}
                for n in full2:
                    rs = [dict(r, 관계유형=lab[(n, int(r["후보"]))]) for r in by_item[n]]
                    if any(r["관계유형"] in GOOD for r in rs):
                        items[n] = rs
                for m in METHODS:
                    out.append(f"| {m} | {who} (세분류 {len(items)}개) | "
                               + " | ".join(f"{accuracy(items, GOOD, m, k):.1%}" for k in KS) + " |")
            out.append("")

    text = "\n".join(out)
    (OCC / f"{prefix}_evaluation.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
