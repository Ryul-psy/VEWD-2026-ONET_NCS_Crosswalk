"""NCS 원자료를 정제해 표 두 개를 만든다.

원자료(data-NCS/NCS-DB(2026-02).xlsx)는 한 행이 "수행준거 × 지식·기술·태도" 조합이라
같은 수행준거와 같은 지식·기술·태도가 여러 번 반복된다.

    python clean_ncs.py

결과 (UTF-8 BOM, 엑셀에서 바로 열림)
    data-NCS/ncs_performance_criteria.csv  수행준거 하나가 한 행
    data-NCS/ncs_ksa_criteria.csv          수행준거·지식·기술·태도 항목 하나가 한 행
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).parent
SOURCE = ROOT / "data-NCS" / "NCS-DB(2026-02).xlsx"
OUTPUT = ROOT / "data-NCS" / "ncs_performance_criteria.csv"
OUTPUT_KSA = ROOT / "data-NCS" / "ncs_ksa_criteria.csv"

# 원자료에서 그대로 가져오는 열 (지식·기술·태도 열 4개는 버린다)
KEEP = [
    "대분류코드", "대분류코드명", "중분류코드", "중분류코드명",
    "소분류코드", "소분류코드명", "세분류코드", "세분류코드명",
    "능력단위분류번호", "능력단위명칭", "수준",
    "능력단위요소번호", "능력단위요소명", "능력단위요소수준",
    "수행준거번호", "수행준거",
]
# 새로 만드는 열
#   수행준거ID      : 능력단위요소번호-수행준거번호 (예: 0101010101_17v2-1-1)
#   NCS세분류코드   : 대·중·소·세분류코드를 이은 8자리
#   수행준거_글자수 : 정제된 수행준거의 글자 수
COLUMNS = ["수행준거ID", "NCS세분류코드"] + KEEP + ["수행준거_글자수"]

# 지식·기술·태도는 능력단위요소에 딸린 항목이다 (수행준거마다 따로 있지 않다)
KSA = ["지식기술태도코드", "지식기술태도코드명", "지식기술태도번호", "지식기술태도의의"]
KSA_LETTER = {"지식": "K", "기술": "S", "태도": "A"}
KIND_ORDER = {"수행준거": 0, "지식": 1, "기술": 2, "태도": 3}
# 두 번째 표의 열
#   항목ID : 수행준거는 수행준거ID와 같고, 지식·기술·태도는 능력단위요소번호-K1, -S1, -A1 꼴
#   구분   : 수행준거, 지식, 기술, 태도
COLUMNS_KSA = [
    "항목ID", "NCS세분류코드", "세분류코드명", "능력단위분류번호", "능력단위명칭",
    "능력단위요소번호", "능력단위요소명", "구분", "번호", "내용", "글자수",
]

SPACES = re.compile(r"\s+")
# 글꼴 전용 기호(사용자 정의 영역), 폭 없는 공백처럼 화면에 보이지 않거나 깨지는 문자
INVISIBLE = re.compile("[\ue000-\uf8ff\u200b-\u200d\ufeff]")
ENDING = re.compile(r"[\s.,]+$")
HAS_LETTER = re.compile(r"[가-힣A-Za-z]")


def clean(value) -> str:
    """셀 값을 문자열로 바꾸고 유니코드(NFC)와 공백을 정리한다."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = unicodedata.normalize("NFC", str(value)).replace("_x000D_", " ")
    text = INVISIBLE.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def clean_statement(text: str) -> str:
    """수행준거 문장 끝의 마침표를 하나로 맞춘다. ("있다", "있다..", "있다 ." → "있다.")"""
    if not HAS_LETTER.search(text):
        return ""  # "."처럼 글자가 없는 값은 문장이 아니다
    return ENDING.sub("", text) + "."


def main() -> int:
    if not SOURCE.exists():
        print(f"NCS 원자료가 없습니다: {SOURCE.relative_to(ROOT)}")
        print("이 파일은 저장소에 없습니다. 따로 공유받아 위 경로에 넣어 주세요.")
        return 1
    workbook = openpyxl.load_workbook(SOURCE, read_only=True)
    seen: dict[str, str] = {}
    listed_under: dict[str, str] = {}
    stats = Counter()
    items: dict[str, tuple] = {}       # 항목ID -> 두 번째 표의 한 행
    element_order: dict[str, int] = {}  # 능력단위요소번호 -> 원자료에서 처음 나온 순서

    def add_item(item_id: str, code: str, row: dict, kind: str, number: str, text: str) -> None:
        order = element_order.setdefault(row["능력단위요소번호"], len(element_order))
        items[item_id] = (
            order, KIND_ORDER[kind], int(number) if number.isdigit() else 0,
            [item_id, code, row["세분류코드명"], row["능력단위분류번호"], row["능력단위명칭"],
             row["능력단위요소번호"], row["능력단위요소명"], kind, number, text, len(text)],
        )

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header = [clean(h) for h in next(rows)]
            missing = [c for c in KEEP + KSA if c not in header]
            if missing:
                print(f"시트 {sheet.title}: 열이 없습니다 {missing}")
                return 1
            index = [header.index(c) for c in KEEP]
            ksa_index = [header.index(c) for c in KSA]

            for raw in rows:
                row = dict(zip(KEEP, (clean(raw[i]) for i in index)))
                if not row["능력단위분류번호"]:
                    stats["빈 행"] += 1
                    continue
                stats["원자료 행"] += 1
                row["수행준거"] = clean_statement(row["수행준거"])
                code = (row["대분류코드"] + row["중분류코드"]
                        + row["소분류코드"] + row["세분류코드"])

                # 지식·기술·태도: 능력단위요소 안에서 종류와 번호가 같으면 한 번만 남긴다
                _, kind, number, text = (clean(raw[i]) for i in ksa_index)
                if kind not in KSA_LETTER or not number or not HAS_LETTER.search(text):
                    stats["지식·기술·태도가 비었거나 글자가 없는 행"] += 1
                else:
                    ksa_id = f"{row['능력단위요소번호'].replace(' ', '-')}-{KSA_LETTER[kind]}{number}"
                    if ksa_id not in items:
                        add_item(ksa_id, code, row, kind, number, text)
                        stats[kind] += 1
                    elif items[ksa_id][3][9] != text:
                        stats["같은 ID인데 내용이 다른 지식·기술·태도 행"] += 1

                if not row["수행준거"] or not row["수행준거번호"]:
                    stats["수행준거가 비었거나 글자가 없는 행"] += 1
                    continue

                pc_id = f"{row['능력단위요소번호'].replace(' ', '-')}-{row['수행준거번호']}"
                if pc_id in seen:
                    if listed_under[pc_id] != code:
                        stats["같은 ID가 다른 세분류에도 실린 행"] += 1
                    if seen[pc_id] != row["수행준거"]:
                        stats["같은 ID인데 문장이 다른 행"] += 1
                    continue
                seen[pc_id] = row["수행준거"]
                listed_under[pc_id] = code
                if not row["능력단위분류번호"].startswith(code):
                    stats["분류코드와 능력단위분류번호가 어긋난 수행준거"] += 1

                writer.writerow([pc_id, code] + [row[c] for c in KEEP]
                                + [len(row["수행준거"])])
                add_item(pc_id, code, row, "수행준거", row["수행준거번호"], row["수행준거"])
                stats["수행준거"] += 1
            print(f"시트 {sheet.title} 완료 (누적 수행준거 {stats['수행준거']:,}개)", flush=True)

    stats["서로 다른 수행준거 문장"] = len(set(seen.values()))

    # 두 번째 표: 능력단위요소별로 수행준거, 지식, 기술, 태도 순서
    with OUTPUT_KSA.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS_KSA)
        for *_, line in sorted(items.values(), key=lambda item: item[:3]):
            writer.writerow(line)
    stats["두 번째 표의 행"] = len(items)
    print()
    for name, count in stats.items():
        print(f"{name}: {count:,}")
    print(f"\n저장: {OUTPUT.relative_to(ROOT)}")
    print(f"저장: {OUTPUT_KSA.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
