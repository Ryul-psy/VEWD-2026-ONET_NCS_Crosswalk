"""세분류 → 직업 연결 가운데 다시 판정할 쌍의 검증표를 만든다.

    python make_occ_recheck.py

두 직업 검증표(occ1, occ2)는 `근접`의 기준이 서로 달랐다. 연계표는 일치·좁음·넓음을 핵심으로 하고,
아래 두 부류만 하나의 기준으로 다시 판정한다.
    불일치    두 검증자가 함께 판정했는데 맞는 연결 여부가 갈린 쌍 (occ2 180쌍, occ1 73쌍)
    근접만    occ2에서 맞는 연결이 `근접`뿐인 세분류의 `근접` 쌍
앞선 판정은 표에 싣지 않고 열쇠 파일에 둔다. 앞선 검증표는 읽기만 한다.

결과: data-ONET-NCS/batch1/occupation/
    occ3_recheck.xlsx            검증자가 `관계유형` 열을 채우는 표
    occ3_recheck_검증자2.xlsx    같은 표 (두 번째 검증자용)
    occ3_key.csv                 쌍마다 출처, 앞선 번호·후보, 두 검증자의 앞선 판정
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from linkdata import ROOT, read_onet, read_onet_table
from make_occ_validation import OCC, RELATIONS

GOOD = {"일치", "좁음", "넓음", "근접"}
STRICT = {"일치", "좁음", "넓음"}
GUIDE = [
    ("다시 판정하는 이유", "앞선 두 직업 검증표에서 `근접`과 `관련`의 경계가 검증자와 검증표에 따라 달랐습니다. "
                    "이 표의 쌍은 아래 기준 하나로 다시 판정합니다. 앞선 판정은 보여 주지 않습니다."),
    ("", ""),
    ("검증 방법", ""),
    ("1", "`검증` 시트에서 세분류(C~F열)와 그 능력단위 목록(G열)을 읽습니다."),
    ("2", "후보 직업의 이름(I열), 설명(J열), 주요 Task(K열)를 읽습니다."),
    ("3", "`관계유형`(L열)을 목록에서 고르고, 겹치는 능력단위나 다른 점을 `메모`(M열)에 구체적으로 적습니다."),
    ("4", "`구분`이 `불일치`인 쌍을 먼저 하고 `근접만`으로 넘어갑니다. 번호 순서대로 진행합니다."),
    ("", ""),
    ("근접과 관련의 경계", "능력단위 목록을 하나씩 대조합니다. 세분류의 능력단위 가운데 절반 이상이 그 직업이 실제로 하는 일"
                    "(설명·주요 Task)에 해당하거나, 직업의 주요 Task 가운데 절반 이상이 세분류의 능력단위에 해당하면 `근접`입니다. "
                    "한두 개만 겹치거나, 분야·대상은 같지만 하는 일이 다르면 `관련`입니다."),
    ("", ""),
    ("관계유형", "판정 기준 (세분류가 다루는 일의 범위와 직업이 다루는 일의 범위를 비교합니다)"),
    ("일치", "세분류와 직업이 같은 범위의 일을 가리킨다. 세분류 하나에 최대 하나만 고릅니다."),
    ("좁음", "세분류가 직업보다 좁다. 세분류의 일이 그 직업 일의 일부다."),
    ("넓음", "세분류가 직업보다 넓다. 그 직업의 일이 세분류 일의 일부다."),
    ("근접", "포함 관계는 아니지만 하는 일이 많이 겹친다 (위 경계 기준)."),
    ("관련", "하는 일의 범위는 다르지만 같은 분야이거나 경력이 이어져 제안이 납득된다. (최종 연계표에서는 제외)"),
    ("오류", "관련이 없다. (최종 연계표에서는 제외)"),
]


def read(name: str) -> dict:
    """(NCS세분류코드, 직업코드) → 그 행의 값."""
    rows = load_workbook(OCC / name, read_only=True)["검증"].iter_rows(values_only=True)
    header = list(next(rows))
    out = {}
    for r in rows:
        if r[0] is not None:
            row = dict(zip(header, r))
            out[(row["NCS세분류코드"], row["O*NET 직업코드"])] = row
    return out


def main() -> int:
    xlsx = OCC / "occ3_recheck.xlsx"
    if xlsx.exists():
        print(f"{xlsx.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다.")
        return 1
    picked = {}     # (세분류, 직업) → (구분, 출처, 행, 검증자1, 검증자2)
    for source in ("occ2", "occ1"):
        first, second = read(f"{source}_validation.xlsx"), read(f"{source}_validation_검증자2.xlsx")
        for pair, row in first.items():
            other = second.get(pair, {}).get("관계유형")
            if other and (row["관계유형"] in GOOD) != (other in GOOD):
                picked[pair] = ("불일치", source, row, row["관계유형"], other)
        if source == "occ2":
            labels = defaultdict(list)
            for (sub, _), row in first.items():
                labels[sub].append(row["관계유형"])
            only_close = {sub for sub, v in labels.items() if "근접" in v and not STRICT & set(v)}
            for pair, row in first.items():
                if pair[0] in only_close and row["관계유형"] == "근접" and pair not in picked:
                    picked[pair] = ("근접만", source, row, row["관계유형"], second.get(pair, {}).get("관계유형") or "")
            print(f"occ2에서 맞는 연결이 `근접`뿐인 세분류: {len(only_close)}개")

    info = {r["O*NET-SOC Code"]: r for r in read_onet_table("Occupation Data")}
    occs = sorted({r["O*NET-SOC Code"] for r in read_onet()})
    by_sub = defaultdict(list)
    for pair, value in picked.items():
        by_sub[(value[0], pair[0])].append((pair, value))
    order = sorted(by_sub, key=lambda k: (k[0] != "불일치", k[1]))

    book = Workbook()
    sheet = book.active
    sheet.title = "검증"
    sheet.append(["번호", "구분", "대분류", "중분류", "소분류", "세분류", "능력단위 목록", "후보", "O*NET 직업",
                  "직업 설명", "주요 Task (중요도 상위 5개)", "관계유형", "메모", "NCS세분류코드", "O*NET 직업코드"])
    shade = PatternFill("solid", start_color="F2F2F2")
    key_rows = []
    for number, group in enumerate(order, 1):
        for cand, (pair, (kind, source, row, one, two)) in enumerate(sorted(by_sub[group], key=lambda x: x[0][1]), 1):
            sheet.append([number, kind, row["대분류"], row["중분류"], row["소분류"], row["세분류"], row["능력단위 목록"], cand,
                          row["O*NET 직업"], row["직업 설명"], row["주요 Task (중요도 상위 5개)"], None, None, pair[0], pair[1]])
            if number % 2 == 0:
                for cell in sheet[sheet.max_row]:
                    cell.fill = shade
            key_rows.append({"번호": number, "후보": cand, "구분": kind, "NCS세분류코드": pair[0], "O*NET 직업코드": pair[1],
                             "출처": source, "앞선_번호": row["번호"], "앞선_후보": row["후보"],
                             "앞선_검증자1": one, "앞선_검증자2": two})
    for column, width in zip("ABCDEFGHIJKLMNO", [6, 8, 14, 14, 16, 18, 50, 6, 28, 50, 60, 10, 30, 12, 12]):
        sheet.column_dimensions[column].width = width
    for r in sheet.iter_rows():
        for cell in r:
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
        guide.append(line)
    guide.column_dimensions["A"].width = 20
    guide.column_dimensions["B"].width = 110
    for r in guide.iter_rows():
        for cell in r:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if r[0].value in ("다시 판정하는 이유", "검증 방법", "근접과 관련의 경계", "관계유형"):
            for cell in r:
                cell.font = Font(bold=True)
    listing = book.create_sheet("직업목록")
    listing.append(["O*NET 직업코드", "O*NET 직업", "직업 설명"])
    for code in occs:
        listing.append([code, info[code]["Title"], info[code]["Description"]])
    for column, width in zip("ABC", (14, 40, 120)):
        listing.column_dimensions[column].width = width
    listing.freeze_panes = "A2"
    book.save(xlsx)
    book.save(OCC / "occ3_recheck_검증자2.xlsx")
    with (OCC / "occ3_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    count = lambda kind: (sum(r["구분"] == kind for r in key_rows), len({r["번호"] for r in key_rows if r["구분"] == kind}))
    print(f"다시 판정할 쌍 {len(key_rows):,}개: 불일치 {count('불일치')[0]}쌍(묶음 {count('불일치')[1]}개), "
          f"근접만 {count('근접만')[0]:,}쌍(세분류 {count('근접만')[1]}개)")
    print(f"저장: {xlsx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
