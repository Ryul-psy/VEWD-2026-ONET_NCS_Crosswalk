"""직업 재판정표(occ3)를 평가하고 세분류 → 직업 연계표를 만든다.

    python evaluate_occ_recheck.py

앞선 검증표와 재판정표는 읽기만 한다.

연계표에 싣는 규칙
    핵심        occ1·occ2에서 검증자 1이 일치·좁음·넓음으로 판정했고 재판정 대상이 아닌 쌍
    재판정 합의  재판정표에서 두 검증자가 모두 맞는 연결(일치·좁음·넓음·근접)로 판정한 쌍
보류 (연계표에 싣지 않고 따로 둔다)
    판정 갈림    재판정표에서 한 검증자만 맞는 연결로 판정한 쌍
    재판정 안 함  검증자 1이 `근접`으로 판정했지만 재판정 대상이 아니었던 쌍

결과
    data-ONET-NCS/batch1/occupation/occ3_evaluation.md
    data-ONET-NCS/crosswalk/occupation/ncs_onet_occupation_crosswalk.csv
    data-ONET-NCS/crosswalk/occupation/ncs_onet_occupation_held.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict

from linkdata import ROOT, read_ncs, read_onet
from make_occ_recheck import GOOD, STRICT, read
from make_occ_validation import OCC

CROSS = ROOT / "data-ONET-NCS" / "crosswalk" / "occupation"
LABELS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
METHODS = ["투표", "이름", "평균", "목록"]


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    one, two = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(one[k] * two[k] for k in one) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def table(header: list, rows: list) -> list[str]:
    return ["| " + " | ".join(map(str, header)) + " |", "|" + " --- |" * len(header)] + \
           ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def main() -> int:
    with (OCC / "occ3_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = {(r["NCS세분류코드"], r["O*NET 직업코드"]): r for r in csv.DictReader(f)}
    one = {(str(p[0]), p[1]): r for p, r in read("occ3_recheck.xlsx").items()}
    two = {(str(p[0]), p[1]): r for p, r in read("occ3_recheck_검증자2.xlsx").items()}
    if not set(one) == set(two) == set(key):
        print("재판정표와 열쇠 파일의 쌍이 다릅니다.")
        return 1
    blank = [p for p in key if one[p]["관계유형"] not in LABELS or two[p]["관계유형"] not in LABELS]
    if blank:
        print(f"판정이 비었거나 목록 밖인 쌍 {len(blank)}개. 평가를 멈춥니다.")
        return 1
    a = {p: one[p]["관계유형"] for p in key}
    b = {p: two[p]["관계유형"] for p in key}
    kinds = ["불일치", "근접만"]
    of = {k: [p for p in key if key[p]["구분"] == k] for k in kinds}
    of["전체"] = list(key)

    out = ["# 직업 재판정표(occ3) 평가", "",
           f"재판정한 쌍 {len(key):,}개: 불일치 {len(of['불일치'])}쌍, 근접만 {len(of['근접만']):,}쌍. "
           "두 검증자가 모든 쌍을 판정했다. 맞는 연결 = 일치·좁음·넓음·근접.", ""]

    out += ["## 1. 판정 분포", ""]
    rows = []
    for k in ("전체", *kinds):
        for name, lab in (("검증자 1", a), ("검증자 2", b)):
            c = Counter(lab[p] for p in of[k])
            good = sum(c[x] for x in GOOD)
            rows.append([k, name, len(of[k])] + [c[x] for x in LABELS] + [f"{good} ({pct(good, len(of[k]))})"])
    out += table(["구분", "검증자", "쌍"] + LABELS + ["맞는 연결"], rows)

    out += ["## 2. 두 검증자의 일치", ""]
    rows = []
    for k in ("전체", *kinds):
        ps = of[k]
        both = sum(a[p] in GOOD and b[p] in GOOD for p in ps)
        neither = sum(a[p] not in GOOD and b[p] not in GOOD for p in ps)
        only1 = sum(a[p] in GOOD and b[p] not in GOOD for p in ps)
        only2 = sum(a[p] not in GOOD and b[p] in GOOD for p in ps)
        rows.append([k, len(ps), both, neither, only1, only2, pct(both + neither, len(ps)),
                     f"{kappa([(a[p] in GOOD, b[p] in GOOD) for p in ps]):.2f}",
                     pct(sum(a[p] == b[p] for p in ps), len(ps)), f"{kappa([(a[p], b[p]) for p in ps]):.2f}"])
    out += table(["구분", "쌍", "둘 다 맞음", "둘 다 아님", "검증자 1만 맞음", "검증자 2만 맞음",
                  "일치율(맞음 여부)", "카파(맞음 여부)", "일치율(여섯 유형)", "카파(여섯 유형)"], rows)
    out += ["검증자 1 (행) × 검증자 2 (열), 전체", ""]
    cross = Counter((a[p], b[p]) for p in key)
    out += table(["검증자 1 \\ 검증자 2"] + LABELS + ["합"],
                 [[x] + [cross[(x, y)] for y in LABELS] + [sum(cross[(x, y)] for y in LABELS)] for x in LABELS])

    out += ["## 3. 앞선 판정과 비교", "",
            "같은 검증자가 같은 쌍에 내린 앞선 판정과 재판정을 맞는 연결 여부로 비교한다. "
            "검증자 2는 앞서 판정한 쌍만 센다.", ""]
    rows = []
    for k in kinds:
        for name, lab, col in (("검증자 1", a, "앞선_검증자1"), ("검증자 2", b, "앞선_검증자2")):
            ps = [p for p in of[k] if key[p][col]]
            c = Counter((key[p][col] in GOOD, lab[p] in GOOD) for p in ps)
            rows.append([k, name, len(ps), c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)],
                         pct(c[(True, False)] + c[(False, True)], len(ps))])
    out += table(["구분", "검증자", "쌍", "맞음→맞음", "맞음→아님", "아님→맞음", "아님→아님", "판정이 바뀐 비율"], rows)

    out += ["## 4. 메모에 적은 겹침과 판정", "",
            "메모에 겹침을 `n/5` 꼴로 적었는지에 따라 나눈다 (대부분 주요 Task 5개 가운데 해당하는 수). "
            "안내의 경계는 절반 이상(3/5 이상)이면 `근접`이다. `3~4/5`처럼 범위로 적은 것은 양 끝이 같은 쪽이면 그쪽에 넣고, "
            "`2~3/5`만 `경계에 걸침`으로 둔다. 능력단위 쪽이 절반 이상이어도 `근접`이므로 3/5 미만인데 맞는 연결인 쌍이 "
            "곧 규칙 위반은 아니다. 검증자 1은 28쌍에만 수를 적었다.", ""]
    rows = []
    for name, lab, src in (("검증자 1", a, one), ("검증자 2", b, two)):
        c = Counter()
        for p in key:
            m = re.search(r"(?<!단위 )(\d)(?:~(\d))?/5(?!\d)", src[p]["메모"] or "")
            low, high = (int(m.group(1)), int(m.group(2) or m.group(1))) if m else (0, 0)
            group = "적지 않음" if not m else "3/5 이상" if low >= 3 else "3/5 미만" if high < 3 else "경계에 걸침"
            c[(group, lab[p] in GOOD)] += 1
        for group in ("3/5 이상", "3/5 미만", "경계에 걸침", "적지 않음"):
            rows.append([name, group, c[(group, True)] + c[(group, False)], c[(group, True)], c[(group, False)]])
    out += table(["검증자", "메모의 겹침", "쌍", "맞는 연결", "맞는 연결 아님"], rows)

    # ---- 연계표 ----
    first, ranks = {}, {}
    for source in ("occ1", "occ2"):
        for p, r in read(f"{source}_validation.xlsx").items():
            first[(str(p[0]), p[1])] = (source, r)
        with (OCC / f"{source}_key.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                ranks[(r["NCS세분류코드"], r["O*NET 직업코드"])] = r
    final, held = [], []

    def line(p, relation, status, second=""):
        source, r = first[p]
        k = ranks.get(p, {})
        return {"NCS세분류코드": p[0], "대분류": r["대분류"], "중분류": r["중분류"], "소분류": r["소분류"], "세분류": r["세분류"],
                "O*NET-SOC 코드": p[1], "O*NET 직업명": r["O*NET 직업"], "검증자1_관계유형": relation, "구분": status,
                "검증자2_관계유형": second, "출처": source, **{f"{m}_순위": k.get(f"{m}_순위", "") for m in METHODS}}

    dropped = 0
    for p, (_, r) in first.items():
        if p in key:
            g1, g2 = a[p] in GOOD, b[p] in GOOD
            if g1 and g2:
                final.append(line(p, a[p], "재판정 합의", b[p]))
            elif g1 or g2:
                held.append(line(p, a[p], "판정 갈림", b[p]))
            else:
                dropped += 1
        elif r["관계유형"] in STRICT:
            final.append(line(p, r["관계유형"], "핵심"))
        elif r["관계유형"] == "근접":
            held.append(line(p, "근접", "재판정 안 함"))
    order = lambda r: (r["NCS세분류코드"], LABELS.index(r["검증자1_관계유형"]), r["O*NET-SOC 코드"])
    final.sort(key=order)
    held.sort(key=order)
    CROSS.mkdir(parents=True, exist_ok=True)
    for name, rows_ in (("ncs_onet_occupation_crosswalk.csv", final), ("ncs_onet_occupation_held.csv", held)):
        with (CROSS / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_[0]))
            writer.writeheader()
            writer.writerows(rows_)

    subs = {r["NCS세분류코드"] for r in read_ncs()}
    occs = {r["O*NET-SOC Code"] for r in read_onet()}
    out += ["## 5. 세분류 → 직업 연계표", "",
            "- **핵심**: 앞선 검증표에서 검증자 1이 일치·좁음·넓음으로 판정했고 재판정 대상이 아닌 쌍.",
            "- **재판정 합의**: 재판정표에서 두 검증자가 모두 맞는 연결로 판정한 쌍. 관계유형은 검증자 1의 것을 쓰고 검증자 2의 것을 옆 열에 둔다.",
            "- **판정 갈림** (보류): 재판정표에서 한 검증자만 맞는 연결로 판정한 쌍. 연계표에는 합의된 쌍만 싣기로 정했다.",
            "- **재판정 안 함** (보류): 검증자 1이 `근접`으로 판정했지만 재판정 대상이 아니었던 쌍. 다시 판정하지 않기로 정했다.",
            "- 주요 Task의 절반 이상이 해당해도 능력단위 1개에만 걸리면 `관련`으로 정했다. 검증자 2가 이 기준으로 판정했고, "
            "판정 갈림 가운데 64쌍이 이 경우다.",
            f"- 재판정에서 두 검증자 모두 맞는 연결이 아니라고 본 {dropped}쌍은 어느 파일에도 싣지 않는다.",
            "- 아래 표의 관계유형은 모두 검증자 1의 판정이다. `판정 갈림`의 `관련`은 검증자 2만 맞는 연결로 본 쌍이다.", ""]
    rows = []
    for name, rows_ in (("연계표", final), ("보류", held)):
        for status in sorted({r["구분"] for r in rows_}):
            part = [r for r in rows_ if r["구분"] == status]
            c = Counter(r["검증자1_관계유형"] for r in part)
            rows.append([name, status, len(part), len({r["NCS세분류코드"] for r in part}),
                         len({r["O*NET-SOC 코드"] for r in part})] + [c[x] for x in LABELS[:5]])
        c = Counter(r["검증자1_관계유형"] for r in rows_)
        rows.append([name, "**합**", len(rows_), len({r["NCS세분류코드"] for r in rows_}),
                     len({r["O*NET-SOC 코드"] for r in rows_})] + [c[x] for x in LABELS[:5]])
    out += table(["파일", "구분", "쌍", "세분류", "직업", "일치", "좁음", "넓음", "근접", "관련"], rows)
    linked = {r["NCS세분류코드"] for r in final}
    only_held = {r["NCS세분류코드"] for r in held} - linked
    rechecked = {p[0] for p in key}
    split = {r["NCS세분류코드"] for r in held if r["구분"] == "판정 갈림"} - linked
    out += [f"- 연결이 있는 세분류: {len(linked):,} / {len(subs):,}개 ({pct(len(linked), len(subs))})",
            f"- 연결이 없는 세분류: {len(subs - linked)}개",
            f"    - 판정 갈림 쌍이 있는 세분류 {len(split)}개",
            f"    - 재판정 안 한 `근접` 쌍만 있는 세분류 {len(only_held - split)}개 (모두 1번 직업 검증표. 재판정 대상을 2번 검증표에서만 뽑았다)",
            f"    - 재판정에서 두 검증자 모두 아니라고 본 세분류 {len(rechecked - linked - only_held)}개",
            f"    - 처음부터 맞는 후보가 없던 세분류 {len(subs - linked - only_held - rechecked)}개",
            f"- 연결이 있는 O*NET 직업: {len({r['O*NET-SOC 코드'] for r in final})} / {len(occs)}개", ""]
    per = Counter(r["NCS세분류코드"] for r in final)
    dist = Counter(min(v, 6) for v in per.values())
    out += table(["세분류당 직업 수"] + [str(k) if k < 6 else "6 이상" for k in range(1, 7)],
                 [["세분류"] + [dist[k] for k in range(1, 7)]])

    # 근접만 세분류의 운명
    out += ["## 6. `근접`뿐이던 세분류", "",
            "2번 직업 검증표에서 검증자 1의 맞는 연결이 `근접`뿐이던 세분류. 이 가운데 7개는 `근접` 쌍이 모두 `불일치` 구분으로 재판정됐다.", ""]
    close_subs = {s for s, v in first_only_close(first).items() if v}
    rows = [["대상 세분류", len(close_subs)],
            ["재판정 뒤 연계표에 연결이 있는 세분류", len(close_subs & linked)],
            ["판정 갈림 쌍만 남은 세분류", len((close_subs - linked) & split)],
            ["두 검증자 모두 맞는 연결이 없다고 본 세분류", len(close_subs - linked - split)]]
    out += table(["항목", "세분류 수"], rows)

    (OCC / "occ3_evaluation.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


def first_only_close(first: dict) -> dict:
    """occ2에서 검증자 1의 맞는 연결이 `근접`뿐이던 세분류."""
    labels = defaultdict(list)
    for p, (source, r) in first.items():
        if source == "occ2":
            labels[p[0]].append(r["관계유형"])
    return {s: "근접" in v and not STRICT & set(v) for s, v in labels.items()}


if __name__ == "__main__":
    sys.exit(main())
