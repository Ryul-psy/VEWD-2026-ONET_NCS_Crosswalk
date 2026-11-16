"""사람이 검증할 묶음을 엑셀 표로 만든다 (METHOD.md 6절).

    python make_validation.py            # 1번 묶음
    python make_validation.py --batch 2  # 다음 묶음 (앞 묶음에 쓴 수행준거는 제외)

NCS 대분류 24개에서 수행준거를 같은 수만큼 뽑고, 모델 3가지가 각각 내놓은 상위 5개 Task를
합쳐서 후보로 보여 준다. 어느 모델이 내놓았는지와 순위는 표에 넣지 않고(판정이 치우치지 않게)
열쇠 파일에 따로 둔다.

결과: data-ONET-NCS/batch1/validation/
    batch1_validation.xlsx   검증자가 `관계유형` 열을 채우는 표
    batch1_key.csv           후보별 모델 순위와 점수 (evaluate_models.py가 읽는다)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from find_candidates import ALL_MODELS as MODELS
from linkdata import OUT, read_ncs, read_onet

PER_MAJOR = 13   # 대분류마다 뽑는 수행준거 수 (24 × 13 = 312)
POOL = 5         # 모델마다 후보로 올리는 상위 Task 수
SEED = 20261115
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
VALID = OUT / "validation"

GUIDE = [
    ("검증 방법", ""),
    ("1", "`검증` 시트에서 수행준거(G열)와 그 상위 정보(C~F열)를 읽습니다."),
    ("2", "후보 Task(J열)와 직업명(I열)을 읽습니다. 후보 순서는 무작위이며 모델 순위와 무관합니다."),
    ("3", "`관계유형`(K열)을 목록에서 고릅니다. 판단 근거나 의문은 `메모`(L열)에 적습니다."),
    ("4", "`구분`이 `시범`인 24개(대분류마다 1개)를 먼저 하고 기준을 점검한 뒤 `본`으로 넘어가면 좋습니다."),
    ("5", "일부만 채워도 됩니다. 후보가 모두 채워진 수행준거만 평가에 씁니다."),
    ("", ""),
    ("관계유형", "판정 기준"),
    ("일치", "수행준거와 Task가 같은 범위의 일을 가리킨다. 수행준거 하나에 최대 하나만 고릅니다."),
    ("좁음", "수행준거가 Task의 일부만 담는다."),
    ("넓음", "수행준거가 Task 전체와 그 이상을 담는다."),
    ("근접", "포함 관계는 아니지만 범위가 많이 겹친다."),
    ("관련", "범위는 다르지만 같은 분야여서 제안이 납득된다. (최종 연계표에서는 제외)"),
    ("오류", "관련이 없다. (최종 연계표에서는 제외)"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    VALID.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()

    # 앞 묶음에서 이미 뽑은 수행준거는 제외
    used = set()
    for b in range(1, args.batch):
        with (VALID / f"batch{b}_key.csv").open(encoding="utf-8-sig", newline="") as f:
            used |= {r["수행준거ID"] for r in csv.DictReader(f)}

    # 대분류별 층화 추출. 대분류마다 첫 번째로 뽑힌 것이 시범 묶음이다.
    rng = random.Random(SEED + args.batch)
    by_major = defaultdict(list)
    for i, r in enumerate(ncs):
        if r["수행준거ID"] not in used:
            by_major[r["대분류코드"]].append(i)
    picked = {major: rng.sample(rows, PER_MAJOR) for major, rows in sorted(by_major.items())}
    pilot = [rows[0] for rows in picked.values()]
    main_part = [i for rows in picked.values() for i in rows[1:]]
    sample = pilot + main_part

    # 후보: 모델별 상위 POOL개를 합친다
    files = {m: np.load(OUT / "candidates" / f"{m}.npz") for m in MODELS}
    top = {m: files[m]["fwd_idx"] for m in MODELS}
    score = {m: files[m]["fwd_score"] for m in MODELS}

    book = Workbook()
    sheet = book.active
    sheet.title = "검증"
    header = ["번호", "구분", "대분류", "세분류", "능력단위", "능력단위요소", "수행준거",
              "후보", "O*NET 직업", "O*NET Task", "관계유형", "메모", "수행준거ID", "Task ID"]
    sheet.append(header)
    key_rows = []
    shade = PatternFill("solid", start_color="F2F2F2")

    for number, i in enumerate(sample, 1):
        r = ncs[i]
        pool = sorted({int(j) for m in MODELS for j in top[m][i, :POOL]})
        rng.shuffle(pool)
        for cand, j in enumerate(pool, 1):
            o = onet[j]
            sheet.append([number, "시범" if number <= len(pilot) else "본", r["대분류코드명"],
                          r["세분류코드명"], r["능력단위명칭"], r["능력단위요소명"], r["수행준거"],
                          cand, o["Title"], o["Task"], None, None, r["수행준거ID"], o["Task ID"]])
            if number % 2 == 0:
                for cell in sheet[sheet.max_row]:
                    cell.fill = shade
            key = {"번호": number, "후보": cand, "수행준거ID": r["수행준거ID"], "Task ID": o["Task ID"]}
            for m in MODELS:
                ranks = top[m][i].tolist()
                key[f"{m}_순위"] = ranks.index(j) + 1 if j in ranks else ""
                key[f"{m}_점수"] = f"{score[m][i, ranks.index(j)]:.4f}" if j in ranks else ""
            key_rows.append(key)

    # 표 모양: 머리글 고정, 줄바꿈, 열 너비, 관계유형 목록
    widths = [6, 6, 14, 18, 24, 24, 48, 6, 28, 60, 10, 24, 22, 9]
    for column, width in zip("ABCDEFGHIJKLMN", widths):
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
    choice.add(f"K2:K{sheet.max_row}")

    guide = book.create_sheet("안내")
    for line in GUIDE:
        guide.append(line)
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 100
    for row in (1, 8):
        for cell in guide[row]:
            cell.font = Font(bold=True)

    xlsx = VALID / f"batch{args.batch}_validation.xlsx"
    book.save(xlsx)
    with (VALID / f"batch{args.batch}_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    per = len(key_rows) / len(sample)
    print(f"수행준거 {len(sample)}개 (시범 {len(pilot)}개 + 본 {len(main_part)}개), "
          f"판정할 쌍 {len(key_rows):,}개 (수행준거당 평균 {per:.1f}개)")
    print(f"저장: {xlsx.relative_to(OUT.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
