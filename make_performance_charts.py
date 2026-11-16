"""모델별 성능을 엑셀 표와 차트로 정리한다.

    python make_performance_charts.py

ESCO–O*NET 보고서의 표 2(모델 설명), 표 3·4(단계별 상위 k 적중률)에 맞춘 구성이다.
수치는 각 묶음의 *_evaluation.md에서 읽는다. 평가 스크립트를 다시 돌린 뒤 이 스크립트를 실행하면 갱신된다.

결과: data-ONET-NCS/model_performance.xlsx
"""

from __future__ import annotations

import re
import sys

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from linkdata import ROOT

DATA = ROOT / "data-ONET-NCS"
OUT = DATA / "model_performance.xlsx"
KS = ["상위 1", "상위 2", "상위 3", "상위 5"]
RAMP = ["86B6EF", "3987E5", "1C5CAB", "0D366B"]     # 한 색상의 밝은 쪽 → 어두운 쪽 (k가 커지는 순서)
FONT = "Arial"
INK, MUTED, GRID = "0B0B0B", "52514E", "E1E0D9"

# (파일, 표 바로 위 제목에 들어 있는 글자)
SOURCES = {
    "b1": ("batch1/validation/batch1_evaluation_recalc.md", "## 1. 판정된 후보 안에서 다시 정렬"),
    "b2": ("validation2/validation2_evaluation.md", "## 2. 상위 k 적중률: 판정된 후보 안에서 다시 정렬"),
    "b2_all": ("validation2/validation2_evaluation.md", "## 3. 전체 수행준거 480개 기준"),
    "b2_strict": ("validation2/validation2_evaluation.md", "## 4. `근접`을 빼고 본 상위 k 적중률"),
    "b3": ("validation3/validation3_evaluation.md", "## 1. 상위 k 적중률"),
    "b3_all": ("validation3/validation3_evaluation.md", "## 2. 전체 수행준거 240개 기준"),
    "b3_strict": ("validation3/validation3_evaluation.md", "## 3. `근접`을 빼고 본 상위 k 적중률"),
    "o1": ("batch1/occupation/occ1_evaluation.md", "## 상위 k 적중률: 맞는 연결"),
    "o1_exact": ("batch1/occupation/occ1_evaluation.md", "## 상위 k 적중률: `일치` 기준"),
    "o2": ("batch1/occupation/occ2_evaluation.md", "## 상위 k 적중률: 맞는 연결"),
    "o2_exact": ("batch1/occupation/occ2_evaluation.md", "## 상위 k 적중률: `일치` 기준"),
}
MODELS = [
    ("M1", "multilingual-e5-large, 방식 A(문장만) 코사인 유사도", "1, 2"),
    ("M3", "0.5 × 방식 A + 0.5 × 방식 B(상위 분류 이름을 붙인 문장)", "1, 2"),
    ("M4", "M1에 쏠림 보정 (CSLS)", "1"),
    ("M5", "M3에 쏠림 보정 (CSLS)", "1, 2"),
    ("IN", "multilingual-e5-large-instruct, 0.5 × A + 0.5 × B, 쏠림 보정", "2, 3"),
    ("RR", "bge-reranker-v2-m3 재순위, A형식과 B형식 점수의 합", "2"),
    ("EN", "표준화한 M5 + IN + RR", "2"),
    ("OC", "z(IN) + z(직업). 직업 = 세분류 → 직업 투표 점수", "3"),
    ("RB", "bge-reranker-v2-m3 재순위, B형식만", "3"),
    ("IB", "z(IN) + z(RB)", "3"),
    ("FN", "z(IN) + z(직업) + z(RB). 최종 모델", "3"),
    ("이름", "세분류 이름(대 > 중 > 소 > 세분류) ↔ 직업명·직업 설명", "직업 1, 2"),
    ("목록", "세분류의 능력단위 이름 목록 ↔ 직업 설명", "직업 1, 2"),
    ("평균", "세분류 수행준거 임베딩의 평균 ↔ 직업 Task 임베딩의 평균", "직업 1, 2"),
    ("투표", "세분류 수행준거마다 찾은 M3 상위 10개 Task의 직업을 세고, 흔한 직업은 깎음. FN의 직업 점수", "직업 1, 2"),
]


def read_table(key: str) -> tuple[list, int]:
    """제목 아래 첫 표의 (모델, 상위 1·2·3·5 비율)과 제목에 적힌 분모."""
    path, heading = SOURCES[key]
    lines = (DATA / path).read_text(encoding="utf-8").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    rows = []
    for l in lines[start + 1:]:
        if l.startswith("## ") or (rows and not l.startswith("|")):
            break
        cells = [c.strip() for c in l.strip("|").split("|")]
        values = [re.match(r"([\d.]+)%", c) for c in cells[1:5]] if len(cells) >= 5 else []
        if len(values) == 4 and all(values):
            rows.append([cells[0]] + [float(v.group(1)) / 100 for v in values])
    if not rows:
        raise ValueError(f"{path}: `{heading}` 아래에서 표를 찾지 못했습니다.")
    rows = [r for r in rows if not r[0].startswith("옛")]      # 1번 묶음의 수정 전 모델은 뺀다
    n = re.search(r"(\d+)개", lines[start]) or re.search(r"맞는 연결이 있는 수행준거 (\d+)개", "\n".join(lines))
    return rows, int(n.group(1))


def style(cell, bold=False, color=INK, size=10, fill=None, wrap=False, center=False):
    cell.font = Font(name=FONT, bold=bold, color=color, size=size)
    cell.alignment = Alignment(wrap_text=wrap, vertical="top" if wrap else "center",
                               horizontal="center" if center else None)
    if fill:
        cell.fill = PatternFill("solid", start_color=fill)


def block(sheet, top: int, title: str, note: str, rows: list, first: str = "모델", highlight: str = "",
          n: int = 0, unit: str = "수행준거", basis: str = "맞는 연결이 있는") -> int:
    """제목, 설명, 표, 차트를 놓고 다음 블록의 시작 행을 돌려준다."""
    style(sheet.cell(top, 1, title), bold=True, size=12)
    style(sheet.cell(top + 1, 1, note), color=MUTED, size=9)
    if n and highlight:
        row = next(r for r in rows if r[0] == highlight)
        who = f"{highlight} {'방식' if unit == '세분류' else '모델'}"
        hit = "`일치` 직업" if "일치` 직업" in basis else "일치·좁음·넓음 연결" if "일치·좁음" in basis else "맞는 연결"
        style(sheet.cell(top + 2, 1,
                         f"읽는 예: {highlight}의 상위 1 {row[1]:.1%}는 {basis} {unit} {n:,}개 가운데 약 {round(row[1] * n):,}개에서 "
                         f"{who}의 1위 후보가 {hit}이었다는 뜻입니다. 상위 5 {row[4]:.1%}는 약 {round(row[4] * n):,}개에서 "
                         f"1~5위 후보 안에 {hit}이 있었다는 뜻입니다."), color=MUTED, size=9)
    head = top + 3
    line = Side(style="thin", color=GRID)
    for c, text in enumerate([first] + KS, 1):
        cell = sheet.cell(head, c, text)
        style(cell, bold=True, fill="F0EFEC", center=c > 1)
        cell.border = Border(bottom=Side(style="thin", color="C3C2B7"))
    for r, row in enumerate(rows, head + 1):
        for c, value in enumerate(row, 1):
            cell = sheet.cell(r, c, value)
            style(cell, bold=row[0] == highlight, center=c > 1)
            cell.border = Border(bottom=line)
            if c > 1:
                cell.number_format = "0.0%"
    last = head + len(rows)

    chart = BarChart()
    chart.type, chart.grouping = "col", "clustered"
    chart.title = title
    chart.add_data(Reference(sheet, min_col=2, max_col=5, min_row=head, max_row=last), titles_from_data=True)
    chart.set_categories(Reference(sheet, min_col=1, min_row=head + 1, max_row=last))
    for series, color in zip(chart.series, RAMP):
        series.graphicalProperties = GraphicalProperties(solidFill=color)
        series.graphicalProperties.line.noFill = True
    chart.gapWidth, chart.overlap = 70, -8
    chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0, 1
    chart.y_axis.majorUnit = 0.2
    chart.y_axis.number_format = "0%"
    chart.y_axis.title = "적중률"
    chart.y_axis.majorGridlines.spPr = GraphicalProperties()
    chart.y_axis.majorGridlines.spPr.line.solidFill = GRID
    chart.x_axis.delete = chart.y_axis.delete = False
    chart.legend.position = "b"
    chart.legend.overlay = False
    chart.title.overlay = chart.y_axis.title.overlay = False
    chart.roundedCorners = False
    chart.width, chart.height = 17, 8.5
    sheet.add_chart(chart, f"G{top}")
    return max(last, top + 17) + 3


def main() -> int:
    data = {k: read_table(k) for k in SOURCES}
    pick = lambda key, model: next(r for r in data[key][0] if r[0] == model)
    book = Workbook()

    # ---- 안내 ----
    guide = book.active
    guide.title = "안내"
    fn, n3 = pick("b3", "FN"), data["b3"][1]
    lines = [
        ("NCS – O*NET 연계: 모델별 성능", ""),
        ("", "ESCO–O*NET 보고서의 표 2(모델 설명)와 표 3·4(단계별 상위 k 적중률)에 맞춰 정리했습니다."),
        ("", ""),
        ("수치의 뜻", ""),
        ("상위 k 적중률", "표와 차트의 모든 백분율입니다. 맞는 연결이 하나라도 있는 항목 가운데, 그 모델이 낸 1~k위 후보 안에 맞는 연결이 "
                       "하나 이상 있던 항목의 비율입니다."),
        ("읽는 예", f"3번 묶음에서 FN의 상위 1은 {fn[1]:.1%}입니다. 맞는 Task가 있는 수행준거 {n3}개 가운데 {round(fn[1] * n3)}개에서 FN이 1위로 낸 "
                  f"Task가 맞는 연결이었다는 뜻입니다. 상위 5는 {fn[4]:.1%}로, {round(fn[4] * n3)}개에서 1~5위 안에 맞는 Task가 있었습니다."),
        ("상위 1", "모델이 1위로 낸 후보가 맞는 연결인 비율. 사람이 확인하지 않고 1위만 쓸 때의 정확도에 해당합니다."),
        ("상위 2, 3, 5", "후보를 2개, 3개, 5개까지 볼 때 그 안에 맞는 연결이 있는 비율. k가 커지면 값은 같거나 커집니다. "
                      "사람이 후보 목록에서 고를 때 맞는 것을 찾을 수 있는 비율에 해당합니다."),
        ("맞는 연결", "검증자가 일치·좁음·넓음·근접으로 판정한 연결. `관련`과 `오류`는 맞는 연결이 아닙니다. 보고서 표 4와 같은 기준입니다."),
        ("분모 (항목 수)", "차트 제목에 적힌 수. 기본은 `맞는 연결이 하나라도 있는 항목`입니다. 후보 어디에도 맞는 연결이 없던 항목은 "
                       "어느 모델도 맞힐 수 없으므로 뺍니다."),
        ("항목", "수행준거 → Task 시트에서는 수행준거 하나, 세분류 → 직업 시트에서는 세분류 하나입니다."),
        ("", ""),
        ("기준이 다른 표", ""),
        ("`근접` 제외", "일치·좁음·넓음만 맞는 연결로 센 값. 범위가 포함 관계로 분명한 연결만 볼 때의 성능입니다. 분모가 46~50개로 작아 "
                     "한 항목이 약 2%p를 움직입니다."),
        ("전체 수행준거 기준", "맞는 Task가 후보에 없던 수행준거까지 분모에 넣은 값. 연계표를 그대로 쓸 때 1위(또는 상위 k)가 맞는 연결일 "
                          "비율에 가깝습니다. 기본 기준보다 낮게 나옵니다."),
        ("`일치` 기준", "`일치`로 판정된 직업이 있는 세분류만 분모로 하고, 그 `일치` 직업이 상위 k 안에 있던 비율. 보고서 표 3(exact match)과 같습니다."),
        ("다시 정렬한 값", "1번 묶음 전체와 2번 묶음의 M1·M3. 그 모델의 상위 후보가 모두 판정되지는 않아, 판정된 후보만 그 모델의 점수로 "
                       "다시 줄 세워 계산했습니다. 나머지는 모델의 순위 그대로 계산한 값입니다."),
        ("", ""),
        ("읽을 때 주의할 점", ""),
        ("같은 묶음 안에서만 비교", "묶음마다 표본과 후보가 다릅니다. 1번 묶음의 43.2%와 3번 묶음의 51.4%는 견줄 수 없습니다."),
        ("직업 1번과 2번", "`근접`의 판정 기준이 달랐습니다. 직업 2번의 80.0%가 직업 1번의 51.8%보다 높은 것은 방식이 나아져서가 아니라 "
                      "판정이 너그러웠기 때문입니다 (METHOD.md 5.4절). 방식의 순서만 봅니다."),
        ("차이의 유의성", "막대 높이가 달라도 우연일 수 있습니다. 각 차트 위 설명에 검정 결과를 적었습니다. p는 두 모델이 실제로는 같을 때 "
                      "이만한 차이가 우연히 나올 확률이고, 0.05보다 작으면 차이가 있다고 봅니다."),
        ("상위 k의 한계", "검증표에 모델별 상위 5개 후보만 실었으므로 k는 5까지만 봅니다. 6위 아래에 맞는 연결이 있는지는 알 수 없습니다."),
        ("", ""),
        ("차트 읽는 법", ""),
        ("막대", "막대 네 개가 한 모델(방식)의 상위 1, 2, 3, 5 적중률입니다. 색이 짙을수록 k가 큽니다. 세로축은 모든 차트가 0~100%입니다."),
        ("굵은 글씨 행", "그 묶음에서 고른 모델입니다."),
        ("", ""),
        ("시트", ""),
        ("단계별 요약", "묶음마다 고른 모델의 성능 (보고서 표 3·4의 꼴)"),
        ("모델", "비교한 모델과 방식의 설명 (보고서 표 2의 꼴)"),
        ("수행준거-Task", "1~3번 묶음의 모델별 상위 k 적중률"),
        ("수행준거-Task 다른 기준", "`근접`을 뺀 기준, 전체 수행준거 기준"),
        ("세분류-직업", "직업 1·2번 검증표의 방식별 상위 k 적중률 (맞는 연결 기준, `일치` 기준)"),
        ("", ""),
        ("출처", "수치는 data-ONET-NCS 아래 각 묶음의 *_evaluation.md에서 읽었습니다. make_performance_charts.py로 다시 만들 수 있습니다. "
               "`약 n개`로 적은 수는 적중률 × 분모를 반올림한 값입니다."),
    ]
    for r, (term, text) in enumerate(lines, 1):
        head = bool(term) and not text
        style(guide.cell(r, 1, term), bold=True, size=14 if r == 1 else 11 if head else 10, wrap=not head)
        style(guide.cell(r, 2, text), wrap=True)
    guide.column_dimensions["A"].width = 24
    guide.column_dimensions["B"].width = 125

    # ---- 단계별 요약 ----
    stage = book.create_sheet("단계별 요약")
    stages = [("1번 묶음", "M5", "b1"), ("2번 묶음", "IN", "b2"), ("3번 묶음", "FN", "b3"),
              ("직업 1번", "투표", "o1"), ("직업 2번", "투표", "o2")]
    rows = [[f"{name} · {model}"] + pick(key, model)[1:] for name, model, key in stages]
    nxt = block(stage, 1, "단계별로 고른 모델의 상위 k 적중률",
                "묶음마다 표본과 후보가 달라 막대 묶음끼리는 견줄 수 없습니다. 앞 셋은 수행준거 → Task, 뒤 둘은 세분류 → 직업입니다.",
                rows, first="단계 · 모델")
    style(stage.cell(nxt, 1, "단계별 표본"), bold=True, size=12)
    for c, text in enumerate(["단계", "고른 모델", "단위", "맞는 연결이 있는 항목 수", "상위 1에 맞는 연결이 있던 항목 수"], 1):
        style(stage.cell(nxt + 1, c, text), bold=True, fill="F0EFEC")
    for r, (name, model, key) in enumerate(stages, nxt + 2):
        n = data[key][1]
        values = [name, model, "세분류" if key.startswith("o") else "수행준거", n, round(pick(key, model)[1] * n)]
        for c, value in enumerate(values, 1):
            style(stage.cell(r, c, value))
    style(stage.cell(nxt + 8, 1, "`맞는 연결이 있는 항목 수`는 적중률의 분모입니다. 상위 1 항목 수는 적중률 × 분모를 반올림한 값입니다. "
                                 "보고서의 모델은 상위 1 적중률이 85~91%였습니다. 본 작업은 단위가 직업이 아니라 문장이고 언어가 달라 낮습니다."),
          color=MUTED, size=9)

    # ---- 모델 ----
    models = book.create_sheet("모델")
    style(models.cell(1, 1, "비교한 모델과 방식"), bold=True, size=12)
    for c, text in enumerate(["이름", "점수 계산", "쓰인 묶음"], 1):
        style(models.cell(3, c, text), bold=True, fill="F0EFEC")
    for r, row in enumerate(MODELS, 4):
        for c, value in enumerate(row, 1):
            style(models.cell(r, c, value), bold=row[0] == "FN", wrap=c == 2)
    note = len(MODELS) + 5
    for r, text in enumerate([
        "방식 A: 수행준거 문장 ↔ Task 문장.  방식 B: `세분류명 > 능력단위명 > 능력단위요소명: 수행준거` ↔ `직업명: Task`.",
        "쏠림 보정(CSLS): 2 × 유사도 − 수행준거의 상위 10개 평균 − Task의 상위 10개 평균. 어디에나 가깝게 나오는 문장을 깎습니다.",
        "z: 수행준거마다 IN 상위 50개 Task 안에서 표준화한 값. 결합 모델의 가중치는 판정 전에 모두 1로 정했습니다.",
        "앞의 11개는 수행준거 → Task 모델, 뒤의 4개는 세분류 → 직업 방식입니다.",
    ], note):
        style(models.cell(r, 1, text), color=MUTED, size=9)
    for column, width in zip("ABC", (10, 95, 14)):
        models.column_dimensions[column].width = width

    # ---- 수행준거 → Task ----
    task = book.create_sheet("수행준거-Task")
    row = block(task, 1, f"1번 묶음: 맞는 연결이 있는 수행준거 {data['b1'][1]}개",
                "multilingual-e5-large 기반 네 모델. 판정된 후보 안에서 다시 정렬한 값. M5가 가장 높아 2번 묶음의 기준이 됐습니다.",
                data["b1"][0], highlight="M5", n=data["b1"][1])
    row = block(task, row, f"2번 묶음: 맞는 연결이 있는 수행준거 {data['b2'][1]}개",
                "IN이 M5보다 나았습니다 (상위 1에서 한쪽만 맞힌 수행준거 30 대 14, 부호검정 p = 0.023). IN을 기본 검색 모델로 정했습니다.",
                data["b2"][0], highlight="IN", n=data["b2"][1])
    block(task, row, f"3번 묶음: 맞는 연결이 있는 수행준거 {data['b3'][1]}개",
          "직업 점수를 넣은 OC와 FN이 나머지 셋보다 나았습니다 (p ≤ 0.001). 모든 k에서 가장 높은 FN이 최종 모델입니다.",
          data["b3"][0], highlight="FN", n=data["b3"][1])

    other = book.create_sheet("수행준거-Task 다른 기준")
    row = block(other, 1, f"2번 묶음, `근접` 제외: 수행준거 {data['b2_strict'][1]}개",
                "일치·좁음·넓음만 맞는 연결로 본 값입니다. 표본이 작습니다.", data["b2_strict"][0], highlight="IN", n=data["b2_strict"][1], basis="일치·좁음·넓음 연결이 있는")
    row = block(other, row, f"3번 묶음, `근접` 제외: 수행준거 {data['b3_strict'][1]}개",
                "일치·좁음·넓음만 맞는 연결로 본 값입니다. 표본이 작습니다.", data["b3_strict"][0], highlight="FN", n=data["b3_strict"][1], basis="일치·좁음·넓음 연결이 있는")
    row = block(other, row, "2번 묶음, 전체 수행준거 480개 기준",
                "맞는 Task가 후보에 없던 수행준거까지 분모에 넣은 값입니다. 연계표를 그대로 쓸 때 기대할 수 있는 수준에 가깝습니다.",
                data["b2_all"][0], highlight="IN", n=480, basis="검증한 전체")
    block(other, row, "3번 묶음, 전체 수행준거 240개 기준",
          "맞는 Task가 후보에 없던 수행준거까지 분모에 넣은 값입니다. FN의 1위가 맞는 연결인 비율은 30.0%입니다.",
          data["b3_all"][0], highlight="FN", n=240, basis="검증한 전체")

    # ---- 세분류 → 직업 ----
    occ = book.create_sheet("세분류-직업")
    row = block(occ, 1, f"직업 1번, 맞는 연결 기준: 세분류 {data['o1'][1]}개",
                "보고서 표 4와 같은 기준입니다. 투표가 가장 높아 FN의 직업 점수로 씁니다.", data["o1"][0], first="방식", highlight="투표", n=data["o1"][1], unit="세분류")
    row = block(occ, row, f"직업 1번, `일치` 기준: 세분류 {data['o1_exact'][1]}개",
                "보고서 표 3(exact match)과 같은 기준입니다. 표본이 작습니다.", data["o1_exact"][0], first="방식", highlight="투표", n=data["o1_exact"][1], unit="세분류", basis="`일치` 직업이 있는")
    row = block(occ, row, f"직업 2번, 맞는 연결 기준: 세분류 {data['o2'][1]}개",
                "`근접`의 판정 기준이 직업 1번과 달라 수치를 견줄 수 없습니다. 방식의 순서만 봅니다.", data["o2"][0], first="방식", highlight="투표", n=data["o2"][1], unit="세분류")
    block(occ, row, f"직업 2번, `일치` 기준: 세분류 {data['o2_exact'][1]}개",
          "보고서 표 3(exact match)과 같은 기준입니다.", data["o2_exact"][0], first="방식", highlight="투표", n=data["o2_exact"][1], unit="세분류", basis="`일치` 직업이 있는")

    for sheet in (stage, task, other, occ):
        sheet.column_dimensions["A"].width = 22 if sheet is stage else 12
        for column in "BCDE":
            sheet.column_dimensions[column].width = 12 if sheet is not stage else 16
        sheet.sheet_view.showGridLines = False
    guide.sheet_view.showGridLines = models.sheet_view.showGridLines = False
    book.save(OUT)
    print(f"저장: {OUT.relative_to(ROOT)}")
    for key, (rows, n) in data.items():
        print(f"  {key}: 분모 {n}, " + ", ".join(f"{r[0]} {r[1]:.1%}/{r[4]:.1%}" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
