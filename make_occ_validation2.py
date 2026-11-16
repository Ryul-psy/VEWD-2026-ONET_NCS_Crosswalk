"""NCS 세분류 → O*NET 직업 검증표를 나머지 세분류 전체(990개)로 넓힌다.

    python make_occ_validation2.py

1번 직업 검증표(occ1, 119개)에 없는 세분류가 대상이다. 후보는 make_occ_validation.py가 저장한
candidates_*.csv에서 가져온다: 투표 상위 5개 + 이름 상위 3개 + 평균 상위 3개.
1번 검증표에서 이 규칙은 네 방식의 상위 5개를 모두 합친 것보다 쌍이 40% 적으면서,
맞는 직업이 있는 세분류 110개 중 106개, 맞는 쌍 204개 중 185개를 담았다.

번호는 대분류를 돌아가며 매긴다. 어디서 멈춰도 대분류가 고르게 들어간다.

결과: data-ONET-NCS/batch1/occupation/
    occ2_validation.xlsx            검증자가 `관계유형` 열을 채우는 표
    occ2_validation_검증자2.xlsx    두 번째 검증자용 (앞 72개 세분류)
    occ2_key.csv                    후보별 방식 순위
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from linkdata import ROOT, read_ncs, read_onet, read_onet_table
from make_occ_validation import GUIDE, OCC, RELATIONS

TAKE = {"투표": 5, "이름": 3, "평균": 3}
METHODS = ["이름", "목록", "평균", "투표"]
RATER2 = 72
SEED = 20261119


def write_book(path, sample, pools, first, units, info, main_tasks, occs) -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = "검증"
    sheet.append(["번호", "구분", "대분류", "중분류", "소분류", "세분류", "능력단위 목록", "후보", "O*NET 직업",
                  "직업 설명", "주요 Task (중요도 상위 5개)", "관계유형", "메모", "더 맞는 직업", "NCS세분류코드", "O*NET 직업코드"])
    shade = PatternFill("solid", start_color="F2F2F2")
    for number, c in enumerate(sample, 1):
        r = first[c]
        for cand, code in enumerate(pools[c], 1):
            sheet.append([number, "본", r["대분류코드명"], r["중분류코드명"], r["소분류코드명"], r["세분류코드명"],
                          ", ".join(units[c]), cand, info[code]["Title"], info[code]["Description"], main_tasks[code],
                          None, None, None, c, code])
            if number % 2 == 0:
                for cell in sheet[sheet.max_row]:
                    cell.fill = shade
    for column, width in zip("ABCDEFGHIJKLMNOP", [6, 6, 14, 14, 16, 18, 50, 6, 28, 50, 60, 10, 24, 20, 12, 12]):
        sheet.column_dimensions[column].width = width
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    choice = DataValidation(type="list", formula1='"' + ",".join(RELATIONS) + '"', allow_blank=True)
    choice.error, choice.errorTitle = "목록에서 골라 주세요.", "관계유형"
    sheet.add_data_validation(choice)
    choice.add(f"L2:L{sheet.max_row}")

    guide = book.create_sheet("안내")
    for line in GUIDE:
        if line[0] == "5":
            line = ("5", "번호 순서대로 진행합니다. 대분류를 돌아가며 번호를 매겼으므로 어디서 멈춰도 됩니다.")
        guide.append(line)
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 110
    for row in (1, 9):
        for cell in guide[row]:
            cell.font = Font(bold=True)
    listing = book.create_sheet("직업목록")
    listing.append(["O*NET 직업코드", "O*NET 직업", "직업 설명"])
    for code in occs:
        listing.append([code, info[code]["Title"], info[code]["Description"]])
    for column, width in zip("ABC", (14, 40, 120)):
        listing.column_dimensions[column].width = width
    for cell in listing[1]:
        cell.font = Font(bold=True)
    listing.freeze_panes = "A2"
    listing.auto_filter.ref = listing.dimensions
    book.save(path)


def main() -> int:
    xlsx = OCC / "occ2_validation.xlsx"
    if xlsx.exists():
        print(f"{xlsx.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다.")
        return 1
    ncs, onet = read_ncs(), read_onet()
    info = {r["O*NET-SOC Code"]: r for r in read_onet_table("Occupation Data")}
    occs = sorted({r["O*NET-SOC Code"] for r in onet})
    first, units = {}, defaultdict(list)
    for r in ncs:
        first.setdefault(r["NCS세분류코드"], r)
        if r["능력단위명칭"] not in units[r["NCS세분류코드"]]:
            units[r["NCS세분류코드"]].append(r["능력단위명칭"])
    importance = {(r["O*NET-SOC Code"], r["Task ID"]): float(r["Data Value"])
                  for r in read_onet_table("Task Ratings") if r["Scale ID"] == "IM"}
    tasks = defaultdict(list)
    for r in onet:
        tasks[r["O*NET-SOC Code"]].append((importance.get((r["O*NET-SOC Code"], r["Task ID"]), 0.0), r["Task"]))
    main_tasks = {c: "\n".join(f"- {t}" for _, t in sorted(v, key=lambda x: -x[0])[:5]) for c, v in tasks.items()}

    ranks = {}
    for m in METHODS:
        with (OCC / f"candidates_{m}.csv").open(encoding="utf-8-sig", newline="") as f:
            ranks[m] = {r["NCS세분류코드"]: [r[f"{k}위_직업코드"] for k in range(1, 11)] for r in csv.DictReader(f)}
    with (OCC / "occ1_key.csv").open(encoding="utf-8-sig", newline="") as f:
        done = {r["NCS세분류코드"] for r in csv.DictReader(f)}

    rng = random.Random(SEED)
    by_major = defaultdict(list)
    for c in sorted(first):
        if c not in done:
            by_major[first[c]["대분류코드"]].append(c)
    for rows in by_major.values():
        rng.shuffle(rows)
    sample = [rows[turn] for turn in range(max(map(len, by_major.values())))
              for _, rows in sorted(by_major.items()) if turn < len(rows)]

    pools, key_rows = {}, []
    for number, c in enumerate(sample, 1):
        pool = sorted({code for m, k in TAKE.items() for code in ranks[m][c][:k]})
        rng.shuffle(pool)
        pools[c] = pool
        for cand, code in enumerate(pool, 1):
            key = {"번호": number, "후보": cand, "NCS세분류코드": c, "O*NET 직업코드": code}
            for m in METHODS:
                key[f"{m}_순위"] = ranks[m][c].index(code) + 1 if code in ranks[m][c] else ""
            key_rows.append(key)

    write_book(xlsx, sample, pools, first, units, info, main_tasks, occs)
    write_book(OCC / "occ2_validation_검증자2.xlsx", sample[:RATER2], pools, first, units, info, main_tasks, occs)
    with (OCC / "occ2_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    second = sum(len(pools[c]) for c in sample[:RATER2])
    print(f"세분류 {len(sample)}개, 판정할 쌍 {len(key_rows):,}개 (세분류당 평균 {len(key_rows) / len(sample):.1f}개)")
    print(f"두 번째 검증자용: 앞 {RATER2}개 세분류, {second}쌍")
    print(f"저장: {xlsx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
