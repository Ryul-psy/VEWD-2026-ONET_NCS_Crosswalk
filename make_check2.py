"""2번 묶음에서 맞는 후보가 없던 수행준거 256개를 두 번째 검증자가 확인하는 표를 만든다.

    python make_check2.py

검증자 1은 이 256개 가운데 156개에 `더 맞는 Task ID`를 직접 찾아 적었고 100개에는 `없음`을 적었다.
    - 156개: 검증자 1이 적은 Task를 보여 주고, 검증자 2가 관계유형을 판정한다.
    - 100개: Task를 보여 주지 않는다. 검증자 2가 `Task목록`에서 찾아 Task ID 또는 `없음`을 적는다.
validation2.xlsx는 읽기만 한다. 검증자 1의 메모는 싣지 않는다.

결과: data-ONET-NCS/validation2/validation2_확인_검증자2.xlsx
"""

from __future__ import annotations

import sys

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from linkdata import ROOT, read_onet

VALID = ROOT / "data-ONET-NCS" / "validation2"
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
GOOD = {"일치", "좁음", "넓음", "근접"}
GUIDE = [
    ("확인 방법", ""),
    ("1", "2번 검증표에서 맞는 후보가 하나도 없던 수행준거 256개입니다. 번호는 2번 검증표의 번호와 같습니다."),
    ("2", "`제안 Task`(I열)가 있는 줄: 다른 검증자가 Task목록에서 직접 찾은 Task입니다. 수행준거와의 관계를 "
          "`관계유형`(J열)에서 고릅니다. 기준은 2번 검증표와 같습니다 (직업 기준)."),
    ("3", "`제안 Task`가 빈 줄: `Task목록` 시트에서 맞는 Task를 찾아 그 Task ID를 `더 맞는 Task ID`(K열)에 적습니다. "
          "찾아봤지만 없으면 `없음`이라고 적습니다."),
    ("4", "제안 Task를 `관련`이나 `오류`로 판정했는데 더 맞는 Task를 알면 K열에 적습니다. 없으면 `없음`."),
    ("5", "판단 근거는 `메모`(L열)에 적습니다."),
    ("", ""),
    ("직업 기준", "Task 문장만 보지 않고 그 Task가 속한 직업(H열)까지 봅니다. 활동이 같아도 직업이 수행준거의 분야(세분류)와 "
              "다르면 `관련`으로 판정합니다. 일치·좁음·넓음·근접은 직업도 맞을 때만 씁니다."),
]


def main() -> int:
    out = VALID / "validation2_확인_검증자2.xlsx"
    if out.exists():
        print(f"{out.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다.")
        return 1
    rows = load_workbook(VALID / "validation2.xlsx", read_only=True)["검증"].iter_rows(values_only=True)
    header = list(next(rows))
    col = {c: header.index(c) for c in header}
    items = {}
    for r in rows:
        if r[col["번호"]] is None:
            continue
        item = items.setdefault(int(r[col["번호"]]), {"row": r, "good": False, "better": None})
        item["good"] |= r[col["관계유형"]] in GOOD
        if r[col["더 맞는 Task ID"]] not in (None, ""):
            item["better"] = str(r[col["더 맞는 Task ID"]]).strip()
    onet = read_onet()
    task = {o["Task ID"]: o for o in onet}

    book = Workbook()
    sheet = book.active
    sheet.title = "확인"
    sheet.append(["번호", "대분류", "세분류", "능력단위", "능력단위요소", "수행준거", "제안 Task ID", "O*NET 직업",
                  "제안 Task", "관계유형", "더 맞는 Task ID", "메모", "수행준거ID"])
    shown = 0
    for n, item in sorted(items.items()):
        if item["good"]:
            continue
        r, o = item["row"], task.get(item["better"])
        shown += o is not None
        sheet.append([n, r[col["대분류"]], r[col["세분류"]], r[col["능력단위"]], r[col["능력단위요소"]], r[col["수행준거"]],
                      o["Task ID"] if o else None, o["Title"] if o else None, o["Task"] if o else None,
                      None, None, None, r[col["수행준거ID"]]])
    for column, width in zip("ABCDEFGHIJKLM", [6, 14, 18, 24, 24, 48, 9, 28, 60, 10, 14, 24, 22]):
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
    choice.add(f"J2:J{sheet.max_row}")

    guide = book.create_sheet("안내")
    for line in GUIDE:
        guide.append(line)
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 110
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if row[0].value in ("확인 방법", "직업 기준"):
            for cell in row:
                cell.font = Font(bold=True)
    listing = book.create_sheet("Task목록")
    listing.append(["Task ID", "O*NET 직업", "O*NET Task"])
    for o in onet:
        listing.append([o["Task ID"], o["Title"], o["Task"]])
    for column, width in zip("ABC", [9, 36, 110]):
        listing.column_dimensions[column].width = width
    listing.freeze_panes = "A2"
    listing.auto_filter.ref = listing.dimensions
    book.save(out)
    total = sheet.max_row - 1
    print(f"수행준거 {total}개: 제안 Task를 판정할 것 {shown}개, 직접 찾을 것 {total - shown}개")
    print(f"저장: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
