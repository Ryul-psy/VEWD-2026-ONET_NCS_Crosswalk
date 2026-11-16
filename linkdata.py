"""연결 작업에서 함께 쓰는 자료 읽기와 입력 문장 만들기 (METHOD.md 2절, 4절)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).parent
NCS_FILE = ROOT / "data-NCS" / "ncs_performance_criteria.csv"
ONET_DIR = ROOT / "data-ONET" / "db_30_2_text"
# 결과 폴더는 임베딩 모델별로 나눈다. batch1 = multilingual-e5-large, batch2 = 재순위 모델 시험
OUT = ROOT / "data-ONET-NCS" / "batch1"
EMB = OUT / "embeddings"

KINDS = ["onet_A", "onet_B", "ncs_A", "ncs_B"]


def read_ncs() -> list[dict]:
    """정제된 NCS 수행준거 (한 행이 수행준거 하나)."""
    with NCS_FILE.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_onet_table(name: str) -> list[dict]:
    """O*NET 30.2 텍스트 표 (탭으로 구분)."""
    with (ONET_DIR / f"{name}.txt").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def read_onet() -> list[dict]:
    """O*NET Task Statements에 직업명(Title)을 붙인 표."""
    titles = {r["O*NET-SOC Code"]: r["Title"] for r in read_onet_table("Occupation Data")}
    tasks = read_onet_table("Task Statements")
    for r in tasks:
        r["Title"] = titles[r["O*NET-SOC Code"]]
    return tasks


def build_texts(kind: str, ncs: list[dict], onet: list[dict]) -> list[str]:
    """방식 A는 문장만, 방식 B는 상위 분류 이름을 앞에 붙인다."""
    if kind == "ncs_A":
        return [r["수행준거"] for r in ncs]
    if kind == "ncs_B":
        return [f'{r["세분류코드명"]} > {r["능력단위명칭"]} > {r["능력단위요소명"]}: {r["수행준거"]}'
                for r in ncs]
    if kind == "onet_A":
        return [r["Task"] for r in onet]
    if kind == "onet_B":
        return [f'{r["Title"]}: {r["Task"]}' for r in onet]
    raise ValueError(kind)
