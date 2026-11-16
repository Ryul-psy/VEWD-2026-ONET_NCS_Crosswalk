"""NCS 세분류 → O*NET 직업 연결의 후보를 만들고, 시범 묶음 검증표를 엑셀로 만든다.

    python make_occ_validation.py

ESCO–O*NET 연계표와 같은 단위(직업 대 직업)로 연결한다. 후보는 네 가지 방식이 각각 내놓은
상위 5개 직업을 합친 것이다. 어느 방식이 내놓았는지와 순위는 표에 넣지 않고 열쇠 파일에 둔다.

    이름    세분류 이름(대>중>소>세분류) ↔ 직업명·직업 설명
    목록    세분류의 능력단위 이름 목록 ↔ 직업 설명
    평균    세분류 수행준거 임베딩의 평균 ↔ 직업 Task 임베딩의 평균
    투표    세분류의 수행준거마다 찾은 Task 상위 10개(M3)가 속한 직업을 세어, 흔한 직업은 깎음

필요한 파일: batch1/embeddings/occ_units.npz (세분류·직업 임베딩. 없으면 새로 계산한다)

결과: data-ONET-NCS/batch1/occupation/
    candidates_{이름,목록,평균,투표}.csv   세분류 1,109개 전부의 상위 10개 직업
    occ1_validation.xlsx                   검증자가 `관계유형` 열을 채우는 표 (시범 묶음)
    occ1_key.csv                           후보별 방식 순위
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from linkdata import EMB, OUT, read_ncs, read_onet, read_onet_table

PER_MAJOR = 5    # 대분류마다 뽑는 세분류 수 (세분류가 그보다 적으면 전부)
POOL = 5         # 방식마다 후보로 올리는 상위 직업 수
K = 10           # 방식마다 저장하는 상위 직업 수
SEED = 20261116
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
OCC = OUT / "occupation"

GUIDE = [
    ("검증 방법", ""),
    ("1", "`검증` 시트에서 세분류(C~F열)와 그 능력단위 목록(G열)을 읽습니다."),
    ("2", "후보 직업의 이름(I열), 설명(J열), 주요 Task(K열)를 읽습니다. 후보 순서는 무작위입니다."),
    ("3", "`관계유형`(L열)을 목록에서 고릅니다. 근거나 의문은 `메모`(M열)에 적습니다."),
    ("4", "후보 가운데 맞는 직업이 없으면 `직업목록` 시트에서 찾아보고, 그 세분류의 첫 행 `더 맞는 직업`(N열)에 "
          "직업 코드나 이름을 적습니다. 찾아도 없으면 `없음`이라고 적습니다."),
    ("5", "`구분`이 `시범`인 세분류(대분류마다 1개)를 먼저 하고 기준을 점검한 뒤 `본`으로 넘어가면 좋습니다."),
    ("6", "일부만 채워도 됩니다. 후보가 모두 채워진 세분류만 평가에 씁니다."),
    ("", ""),
    ("관계유형", "판정 기준 (세분류가 다루는 일의 범위와 직업이 다루는 일의 범위를 비교합니다)"),
    ("일치", "세분류와 직업이 같은 범위의 일을 가리킨다. 세분류 하나에 최대 하나만 고릅니다."),
    ("좁음", "세분류가 직업보다 좁다. 세분류의 일이 그 직업 일의 일부다."),
    ("넓음", "세분류가 직업보다 넓다. 그 직업의 일이 세분류 일의 일부다."),
    ("근접", "포함 관계는 아니지만 하는 일이 많이 겹친다."),
    ("관련", "하는 일의 범위는 다르지만 같은 분야이거나 경력이 이어져 제안이 납득된다. (최종 연계표에서는 제외)"),
    ("오류", "관련이 없다. (최종 연계표에서는 제외)"),
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def zrow(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)


def main() -> int:
    OCC.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()
    info = {r["O*NET-SOC Code"]: r for r in read_onet_table("Occupation Data")}
    occs = sorted({r["O*NET-SOC Code"] for r in onet})        # Task가 있는 직업 923개
    subs = sorted({r["NCS세분류코드"] for r in ncs})          # 세분류 1,109개
    opos = {c: k for k, c in enumerate(occs)}
    spos = {c: k for k, c in enumerate(subs)}
    task_occ = np.array([opos[r["O*NET-SOC Code"]] for r in onet])
    crit_sub = np.array([spos[r["NCS세분류코드"]] for r in ncs])
    first, units = {}, defaultdict(list)
    for r in ncs:
        first.setdefault(r["NCS세분류코드"], r)
        if r["능력단위명칭"] not in units[r["NCS세분류코드"]]:
            units[r["NCS세분류코드"]].append(r["능력단위명칭"])

    # 세분류·직업 임베딩 (이름, 능력단위 목록, 직업명, 직업 설명)
    cache = EMB / "occ_units.npz"
    if cache.exists():
        e = dict(np.load(cache))
    else:
        from e5_embedder import embed, load_model
        model = load_model()
        e = {
            "sub_name": embed(model, [f'{first[c]["대분류코드명"]} > {first[c]["중분류코드명"]} > '
                                      f'{first[c]["소분류코드명"]} > {first[c]["세분류코드명"]}' for c in subs], batch_size=32),
            "sub_units": embed(model, [f'{first[c]["세분류코드명"]}: ' + ", ".join(units[c]) for c in subs], batch_size=16),
            "occ_title": embed(model, [info[c]["Title"] for c in occs], batch_size=32),
            "occ_desc": embed(model, [f'{info[c]["Title"]}: {info[c]["Description"]}' for c in occs], batch_size=16),
        }
        np.savez(cache, **e)
    if len(e["sub_name"]) != len(subs) or len(e["occ_title"]) != len(occs):
        print("저장된 세분류·직업 임베딩의 행 수가 자료와 다릅니다. occ_units.npz를 지우고 다시 실행하세요.")
        return 1

    # 평균 방식: 수행준거·Task 임베딩(방식 A)의 평균끼리 비교
    ncs_A = np.load(EMB / "ncs_A.npy", mmap_mode="r")
    onet_A = np.load(EMB / "onet_A.npy").astype(np.float32)
    sub_mean = np.zeros((len(subs), ncs_A.shape[1]), np.float32)
    for s in range(0, len(ncs), 20000):
        np.add.at(sub_mean, crit_sub[s:s + 20000], np.asarray(ncs_A[s:s + 20000], np.float32))
    occ_mean = np.zeros((len(occs), onet_A.shape[1]), np.float32)
    np.add.at(occ_mean, task_occ, onet_A)

    # 투표 방식: M3 상위 10개 Task의 직업을 세고, 어디서나 자주 나오는 직업은 깎는다
    top = np.load(OUT / "candidates" / "M3.npz")["fwd_idx"]
    votes = np.zeros((len(subs), len(occs)), np.float64)
    np.add.at(votes, (np.repeat(crit_sub, top.shape[1]), task_occ[top].ravel()), 1)
    share = votes / votes.sum(1, keepdims=True)
    lift = share / (votes.sum(0, keepdims=True) / votes.sum())
    vote_score = votes * np.log(np.maximum(lift, 1.0))

    scores = {
        "이름": zrow(e["sub_name"] @ e["occ_title"].T) + zrow(e["sub_name"] @ e["occ_desc"].T),
        "목록": e["sub_units"] @ e["occ_desc"].T,
        "평균": unit(sub_mean) @ unit(occ_mean).T,
        "투표": vote_score,
    }
    order = {m: np.argsort(-s, axis=1, kind="stable")[:, :K] for m, s in scores.items()}
    for m in scores:
        with (OCC / f"candidates_{m}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["NCS세분류코드", "세분류"] + [f"{k}위_{c}" for k in range(1, K + 1) for c in ("직업코드", "직업명")])
            for c in subs:
                row = [c, first[c]["세분류코드명"]]
                for j in order[m][spos[c]]:
                    row += [occs[j], info[occs[j]]["Title"]]
                writer.writerow(row)

    # 직업마다 중요도가 높은 Task 5개 (검증자가 직업의 범위를 가늠하도록)
    importance = {(r["O*NET-SOC Code"], r["Task ID"]): float(r["Data Value"])
                  for r in read_onet_table("Task Ratings") if r["Scale ID"] == "IM"}
    tasks = defaultdict(list)
    for r in onet:
        tasks[r["O*NET-SOC Code"]].append((importance.get((r["O*NET-SOC Code"], r["Task ID"]), 0.0), r["Task"]))
    main_tasks = {c: "\n".join(f"- {t}" for _, t in sorted(v, key=lambda x: -x[0])[:5]) for c, v in tasks.items()}

    # 대분류별 층화 추출. 대분류마다 첫 번째로 뽑힌 것이 시범 묶음이다.
    rng = random.Random(SEED)
    by_major = defaultdict(list)
    for c in subs:
        by_major[first[c]["대분류코드"]].append(c)
    picked = {major: rng.sample(rows, min(PER_MAJOR, len(rows))) for major, rows in sorted(by_major.items())}
    pilot = [rows[0] for rows in picked.values()]
    sample = pilot + [c for rows in picked.values() for c in rows[1:]]

    book = Workbook()
    sheet = book.active
    sheet.title = "검증"
    sheet.append(["번호", "구분", "대분류", "중분류", "소분류", "세분류", "능력단위 목록", "후보", "O*NET 직업",
                  "직업 설명", "주요 Task (중요도 상위 5개)", "관계유형", "메모", "더 맞는 직업", "NCS세분류코드", "O*NET 직업코드"])
    shade = PatternFill("solid", start_color="F2F2F2")
    key_rows = []
    for number, c in enumerate(sample, 1):
        r, i = first[c], spos[c]
        pool = sorted({int(j) for m in scores for j in order[m][i, :POOL]})
        rng.shuffle(pool)
        for cand, j in enumerate(pool, 1):
            code = occs[j]
            sheet.append([number, "시범" if number <= len(pilot) else "본", r["대분류코드명"], r["중분류코드명"],
                          r["소분류코드명"], r["세분류코드명"], ", ".join(units[c]), cand, info[code]["Title"],
                          info[code]["Description"], main_tasks[code], None, None, None, c, code])
            if number % 2 == 0:
                for cell in sheet[sheet.max_row]:
                    cell.fill = shade
            key = {"번호": number, "후보": cand, "NCS세분류코드": c, "O*NET 직업코드": code}
            for m in scores:
                ranks = order[m][i].tolist()
                key[f"{m}_순위"] = ranks.index(j) + 1 if j in ranks else ""
            key_rows.append(key)

    widths = [6, 6, 14, 14, 16, 18, 50, 6, 28, 50, 60, 10, 24, 20, 12, 12]
    for column, width in zip("ABCDEFGHIJKLMNOP", widths):
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

    xlsx = OCC / "occ1_validation.xlsx"
    if xlsx.exists():
        print(f"{xlsx.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다. 새로 만들려면 파일을 옮기거나 지우세요.")
        return 1
    book.save(xlsx)
    with (OCC / "occ1_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"세분류 {len(sample)}개 (시범 {len(pilot)}개 + 본 {len(sample) - len(pilot)}개), "
          f"판정할 쌍 {len(key_rows):,}개 (세분류당 평균 {len(key_rows) / len(sample):.1f}개)")
    overlap = {(a, b): np.mean([len(set(x) & set(y)) for x, y in zip(order[a][:, :POOL].tolist(), order[b][:, :POOL].tolist())])
               for k, a in enumerate(scores) for b in list(scores)[k + 1:]}
    print("방식 사이에 겹치는 상위 5개 직업 수(평균):", {f"{a}-{b}": round(v, 1) for (a, b), v in overlap.items()})
    print(f"저장: {xlsx.relative_to(OUT.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
