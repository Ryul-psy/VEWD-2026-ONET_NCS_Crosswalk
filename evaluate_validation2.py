"""두 번째 수행준거 검증표로 방법 여섯 가지를 비교하고, 두 검증자의 일치도를 본다.

    python evaluate_validation2.py

검증표는 읽기만 한다. 일치·좁음·넓음·근접을 "맞는 연결"로 본다.

후보는 M5·IN·RR·EN의 상위 5개를 합친 것이다. 그래서
    - M5·IN·RR·EN의 상위 5 이내 적중률은 정확하다 (상위 5개가 모두 판정됨).
    - M1·M3는 자기 상위 5개가 모두 판정된 것이 아니다. 두 가지로 본다.
        1. 자기 순위 그대로 (판정되지 않은 Task는 틀린 것으로 센 하한)
        2. 판정된 후보 안에서 다시 정렬 (모든 방법이 같은 후보를 놓고 겨룬다)
동점은 불리한 쪽으로 처리한다 (맞는 연결이 뒤).

결과: data-ONET-NCS/validation2/validation2_evaluation.md
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from math import comb

from openpyxl import load_workbook

from linkdata import OUT

VALID = OUT.parent / "validation2"
METHODS = ["M1", "M3", "M5", "IN", "RR", "EN"]
OWN = ["M5", "IN", "RR", "EN"]          # 상위 5개가 모두 판정된 방법
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
GOOD = {"일치", "좁음", "넓음", "근접"}
KS = [1, 2, 3, 5]


def read_sheet(name: str):
    """(번호, 후보) → 관계유형, 번호 → 대분류, 번호 → `더 맞는 Task ID`에 적힌 값들."""
    rows = load_workbook(VALID / name, read_only=True)["검증"].iter_rows(values_only=True)
    header = list(next(rows))
    col = {c: header.index(c) for c in ("번호", "후보", "관계유형", "더 맞는 Task ID", "대분류")}
    label, major, better = {}, {}, defaultdict(list)
    for r in rows:
        if r[col["번호"]] is None:
            continue
        n = int(r[col["번호"]])
        value = r[col["관계유형"]]
        label[(n, int(r[col["후보"]]))] = value.strip() if isinstance(value, str) else ""
        major[n] = r[col["대분류"]]
        if r[col["더 맞는 Task ID"]] not in (None, ""):
            better[n].append(str(r[col["더 맞는 Task ID"]]).strip())
    return label, major, better


def sign_p(win: int, lose: int) -> float:
    n, k = win + lose, min(win, lose)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum(left[c] * right[c] for c in left) / n ** 2
    return (agree - chance) / (1 - chance)


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main() -> int:
    label, major, better = read_sheet("validation2.xlsx")
    unknown = Counter(v for v in label.values() if v and v not in RELATIONS)
    if unknown:
        print(f"목록에 없는 관계유형이 있습니다: {dict(unknown)}")
        return 1
    with (VALID / "validation2_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = list(csv.DictReader(f))
    by_item = defaultdict(list)
    for r in key:
        r["번호"], r["후보"] = int(r["번호"]), int(r["후보"])
        for m in METHODS:
            r[f"{m}_순위"], r[f"{m}_점수"] = int(r[f"{m}_순위"]), float(r[f"{m}_점수"])
        by_item[r["번호"]].append(r)

    def pool_rank(rs, lab):
        """판정된 후보 안에서 방법별로 다시 매긴 순위 (동점이면 맞는 연결이 뒤)."""
        out = {}
        for m in METHODS:
            # 점수는 소수 넷째 자리로 반올림되어 있으므로 원래 순위로 정렬한다
            order = sorted(rs, key=lambda r: (r[f"{m}_순위"], lab[(r["번호"], r["후보"])] in GOOD))
            for place, r in enumerate(order, 1):
                out[(m, r["후보"])] = place
        return out

    def prepare(lab, numbers):
        items = {}
        for n in numbers:
            rs = by_item[n]
            if not all(lab.get((n, r["후보"])) for r in rs):
                continue
            pr = pool_rank(rs, lab)
            items[n] = [dict(r, 관계유형=lab[(n, r["후보"])], **{f"{m}_후보안": pr[(m, r["후보"])] for m in METHODS})
                        for r in rs]
        return items

    def hit(rs, good, column, k):
        return any(r[column] <= k and r["관계유형"] in good for r in rs)

    def accuracy(items, good, column, k):
        return sum(hit(rs, good, column, k) for rs in items.values()) / len(items) if items else float("nan")

    def top_table(items, good, suffix, methods):
        rows = ["| 방법 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- |" + " --- |" * len(KS)]
        for m in methods:
            rows.append(f"| {m} | " + " | ".join(f"{accuracy(items, good, m + suffix, k):.1%}" for k in KS) + " |")
        return rows + [""]

    done = prepare(label, sorted(by_item))
    if not done:
        print("후보가 모두 판정된 수행준거가 아직 없습니다.")
        return 1
    matched = {n: rs for n, rs in done.items() if any(r["관계유형"] in GOOD for r in rs)}
    strict = {n: rs for n, rs in done.items() if any(r["관계유형"] in {"일치", "좁음", "넓음"} for r in rs)}
    judged = sum(len(rs) for rs in done.values())
    counts = Counter(r["관계유형"] for rs in done.values() for r in rs)
    unmatched = [n for n in done if n not in matched]
    wrote_id = [n for n in unmatched if any(v != "없음" for v in better.get(n, []))]
    wrote_none = [n for n in unmatched if better.get(n) and all(v == "없음" for v in better[n])]

    out = ["# 수행준거 → Task 두 번째 검증 결과", "",
           f"- 판정된 쌍: {judged:,}개 / {len(key):,}개, 후보가 모두 판정된 수행준거: {len(done)}개 / {len(by_item)}개",
           f"- 맞는 연결(일치·좁음·넓음·근접)이 있는 수행준거: {len(matched)}개 ({len(matched) / len(done):.1%})",
           f"- `근접`을 빼고 일치·좁음·넓음이 있는 수행준거: {len(strict)}개 ({len(strict) / len(done):.1%})",
           f"- 맞는 연결이 없는 수행준거: {len(unmatched)}개. 그중 `더 맞는 Task ID`에 "
           f"Task를 적은 것 {len(wrote_id)}개, `없음` {len(wrote_none)}개, 빈칸 {len(unmatched) - len(wrote_id) - len(wrote_none)}개",
           f"- 맞는 연결이 있는 수행준거에도 `더 맞는 Task ID`를 적은 것: "
           f"{sum(any(v != '없음' for v in better.get(n, [])) for n in matched)}개", "",
           "## 관계유형 분포", "", "| 관계유형 | 쌍 수 | 비율 |", "| --- | --- | --- |"]
    out += [f"| {name} | {counts[name]:,} | {counts[name] / judged:.1%} |" for name in RELATIONS] + [""]

    out += [f"## 1. 상위 k 적중률: 자기 순위 그대로 (맞는 연결이 있는 수행준거 {len(matched)}개)", "",
            "M5·IN·RR·EN은 정확한 값, M1·M3는 하한이다.", ""]
    out += top_table(matched, GOOD, "_순위", METHODS)
    out += [f"## 2. 상위 k 적중률: 판정된 후보 안에서 다시 정렬 (수행준거 {len(matched)}개)", ""]
    out += top_table(matched, GOOD, "_후보안", METHODS)
    out += [f"## 3. 전체 수행준거 {len(done)}개 기준: 상위 k 안에 맞는 연결이 있는 비율 (자기 순위)", "",
            "맞는 Task가 없는 수행준거까지 넣은 값이다. 연결표를 그대로 썼을 때 기대할 수 있는 수준.", ""]
    out += top_table(done, GOOD, "_순위", METHODS)
    out += [f"## 4. `근접`을 빼고 본 상위 k 적중률 (자기 순위, 수행준거 {len(strict)}개)", ""]
    out += top_table(strict, {"일치", "좁음", "넓음"}, "_순위", METHODS)

    out += ["## 5. 방법끼리 비교: 상위 1에서 한쪽만 맞힌 수행준거 수 (양측 부호검정)", "",
            "M1·M3가 들어간 비교는 판정된 후보 안 순위, 나머지는 자기 순위로 센다.", "",
            "| 비교 | 앞 방법만 맞힘 | 뒤 방법만 맞힘 | p |", "| --- | --- | --- | --- |"]
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            suffix = "_순위" if a in OWN and b in OWN else "_후보안"
            win = sum(hit(rs, GOOD, a + suffix, 1) and not hit(rs, GOOD, b + suffix, 1) for rs in matched.values())
            lose = sum(hit(rs, GOOD, b + suffix, 1) and not hit(rs, GOOD, a + suffix, 1) for rs in matched.values())
            out.append(f"| {a} 대 {b} | {win} | {lose} | {sign_p(win, lose):.3f} |")
    out.append("")

    out += ["## 6. 방법별 1위 후보의 관계유형 (자기 순위, 전체 수행준거)", "",
            "| 방법 | " + " | ".join(RELATIONS) + " | 판정 안 됨 |", "| --- |" + " --- |" * 7]
    for m in METHODS:
        first = Counter(r["관계유형"] for rs in done.values() for r in rs if r[f"{m}_순위"] == 1)
        none = sum(not any(r[f"{m}_순위"] == 1 for r in rs) for rs in done.values())
        out.append(f"| {m} | " + " | ".join(str(first[name]) for name in RELATIONS) + f" | {none} |")
    out.append("")

    good_rows = [r for rs in done.values() for r in rs if r["관계유형"] in GOOD]
    out += ["## 7. 맞는 연결이 어느 방법의 상위 5개에서 나왔나", "",
            "| 방법 | 상위 5개에 든 맞는 쌍 | 이 방법에서만 나온 맞는 쌍 | 상위 5개 중 맞는 쌍 비율 |", "| --- | --- | --- | --- |"]
    for m in OWN:
        own = [r for r in good_rows if r[f"{m}_순위"] <= 5]
        only = [r for r in own if not any(r[f"{o}_순위"] <= 5 for o in OWN if o != m)]
        out.append(f"| {m} | {len(own)} / {len(good_rows)} | {len(only)} | {len(own) / (5 * len(done)):.1%} |")
    out.append("")

    out += ["## 8. 1위 점수로 `맞는 Task가 없는 수행준거`를 가려낼 수 있나", "",
            "AUC = 맞는 연결이 있는 수행준거의 1위 점수가 없는 수행준거의 1위 점수보다 높을 확률 (0.5면 구분 못 함).", "",
            "| 방법 | AUC: 맞는 연결이 있는가 | AUC: 1위가 맞는가 |", "| --- | --- | --- |"]
    for m in OWN:
        top = {n: next(r for r in rs if r[f"{m}_순위"] == 1) for n, rs in done.items()}
        has = auc([top[n][f"{m}_점수"] for n in matched], [top[n][f"{m}_점수"] for n in unmatched])
        right = auc([r[f"{m}_점수"] for r in top.values() if r["관계유형"] in GOOD],
                    [r[f"{m}_점수"] for r in top.values() if r["관계유형"] not in GOOD])
        out.append(f"| {m} | {has:.2f} | {right:.2f} |")
    out.append("")

    by_major = defaultdict(list)
    for n, rs in done.items():
        by_major[major[n]].append((n in matched, hit(rs, GOOD, "EN_순위", 1), hit(rs, GOOD, "M5_순위", 1)))
    out += ["## 9. 대분류별", "", "| 대분류 | 수행준거 | 맞는 연결 있음 | EN 1위 적중 | M5 1위 적중 |", "| --- | --- | --- | --- | --- |"]
    for name, v in sorted(by_major.items(), key=lambda x: -sum(a for a, _, _ in x[1])):
        out.append(f"| {name} | {len(v)} | {sum(a for a, _, _ in v)} | {sum(b for _, b, _ in v)} | {sum(c for _, _, c in v)} |")
    out.append("")

    # 두 검증자 일치도
    second = VALID / "validation2_검증자2.xlsx"
    if second.exists():
        label2, _, better2 = read_sheet(second.name)
        both = [(label[k], v) for k, v in label2.items() if v and label.get(k)]
        if both:
            binary = [(a in GOOD, b in GOOD) for a, b in both]
            shared = sorted({n for n, _ in label2} & set(done))
            items1, items2 = prepare(label, shared), prepare(label2, shared)
            shared = [n for n in shared if n in items1 and n in items2]
            has1 = {n for n in shared if any(r["관계유형"] in GOOD for r in items1[n])}
            has2 = {n for n in shared if any(r["관계유형"] in GOOD for r in items2[n])}
            out += ["## 10. 두 검증자 일치도", "",
                    f"- 두 사람이 모두 판정한 쌍: {len(both):,}개 (수행준거 {len(shared)}개)",
                    f"- 여섯 유형이 같은 쌍: {sum(a == b for a, b in both) / len(both):.1%} (카파 {kappa(both):.2f})",
                    f"- 맞는 연결 여부가 같은 쌍: {sum(a == b for a, b in binary) / len(binary):.1%} (카파 {kappa(binary):.2f})",
                    f"- 맞는 쌍 수: 검증자 1 {sum(a for a, _ in binary)}개, 검증자 2 {sum(b for _, b in binary)}개, "
                    f"둘 다 {sum(a and b for a, b in binary)}개",
                    f"- 맞는 연결이 있는 수행준거: 검증자 1 {len(has1)}개, 검증자 2 {len(has2)}개, 둘 다 {len(has1 & has2)}개 / {len(shared)}개",
                    f"- `더 맞는 Task ID`를 적은 수행준거(없음 제외): 검증자 1 "
                    f"{sum(any(v != '없음' for v in better.get(n, [])) for n in shared)}개, 검증자 2 "
                    f"{sum(any(v != '없음' for v in better2.get(n, [])) for n in shared)}개", "",
                    "행 = 검증자 1, 열 = 검증자 2", "", "| | " + " | ".join(RELATIONS) + " |", "| --- |" + " --- |" * 6]
            table = Counter(both)
            out += [f"| {a} | " + " | ".join(str(table[(a, b)]) for b in RELATIONS) + " |" for a in RELATIONS]
            out += ["", "### 검증자별 상위 k 적중률 (같은 수행준거, 자기 순위)", "",
                    "| 방법 | 검증자 | " + " | ".join(f"상위 {k}" for k in KS) + " |", "| --- | --- |" + " --- |" * len(KS)]
            for who, items, has in (("1", items1, has1), ("2", items2, has2)):
                sub = {n: items[n] for n in has}
                for m in METHODS:
                    out.append(f"| {m} | {who} (수행준거 {len(sub)}개) | "
                               + " | ".join(f"{accuracy(sub, GOOD, m + '_순위', k):.1%}" for k in KS) + " |")
            out.append("")

    text = "\n".join(out)
    (VALID / "validation2_evaluation.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
