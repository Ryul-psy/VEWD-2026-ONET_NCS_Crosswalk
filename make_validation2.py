"""두 번째 수행준거 검증표를 만든다 (수정된 e5 + e5-instruct + 재순위 모델).

    python make_validation2.py

1번 묶음과 다른 점
    - 수행준거 480개 (대분류 24개 × 20개). 1번 묶음에 쓴 수행준거는 뺀다.
      번호 순서대로 24개씩이 대분류를 한 바퀴 돈다. 어디서 멈춰도 대분류가 고르게 들어간다.
    - 후보는 네 가지 방법의 상위 5개를 합친 것이다.
        M5  multilingual-e5-large, 0.5 × 방식 A + 0.5 × 방식 B, 쏠림 보정
        IN  multilingual-e5-large-instruct, 같은 방식 (수행준거 쪽에만 지시문을 붙인다)
        RR  bge-reranker-v2-m3 재순위 (M5와 IN의 상위 20개를 합친 후보 안에서)
        EN  M5 + IN + RR 표준화 점수의 합 (같은 후보 안에서)
    - 판정 기준: 직업 기준. 활동이 같아도 Task가 속한 직업이 수행준거의 분야와 다르면 `관련`.

IN의 쏠림 보정에서 Task 쪽 항은 수행준거 20,000개 표본으로 추정한다 (전체를 임베딩하지 않기 위해).
중간 결과를 저장하므로 멈춰도 다시 실행하면 이어서 한다.

결과
    data-ONET-NCS/batch3/embeddings/     e5-instruct 임베딩 (batch3 = multilingual-e5-large-instruct)
    data-ONET-NCS/validation2/
        validation2.xlsx            검증자가 `관계유형` 열을 채우는 표
        validation2_검증자2.xlsx    두 번째 검증자용 (앞 72개 수행준거, 같은 번호·후보)
        validation2_key.csv         후보별 방법 순위와 점수
"""

from __future__ import annotations

import csv
import gc
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from sentence_transformers import CrossEncoder, SentenceTransformer

from e5_embedder import pick_device
from linkdata import EMB, OUT, ROOT, build_texts, read_ncs, read_onet

PER_MAJOR = 20
POOL = 5          # 방법마다 후보로 올리는 상위 Task 수
SHORT = 20        # 재순위에 넘기는 M5·IN 상위 Task 수
REF = 20000       # IN 쏠림 보정용 수행준거 표본
K = 10
RATER2 = 72       # 두 번째 검증자가 보는 수행준거 수 (대분류 세 바퀴)
SEED = 20261117
RELATIONS = ["일치", "좁음", "넓음", "근접", "관련", "오류"]
INSTRUCT = "intfloat/multilingual-e5-large-instruct"
QUERY = "Instruct: Retrieve semantically similar text.\nQuery: "
RERANKER = "BAAI/bge-reranker-v2-m3"
EMB3 = ROOT / "data-ONET-NCS" / "batch3" / "embeddings"
VALID = ROOT / "data-ONET-NCS" / "validation2"
WORK = VALID / "work"

GUIDE = [
    ("검증 방법", ""),
    ("1", "`검증` 시트에서 수행준거(G열)와 그 상위 정보(C~F열)를 읽습니다."),
    ("2", "후보 Task(J열)와 직업명(I열)을 읽습니다. 후보 순서는 무작위이며 모델 순위와 무관합니다."),
    ("3", "`관계유형`(K열)을 목록에서 고릅니다. 판단 근거나 의문은 `메모`(L열)에 적습니다."),
    ("4", "맞는 후보가 없는데 더 맞는 Task를 알면 `Task목록` 시트에서 찾아 그 Task ID를 `더 맞는 Task ID`(M열)에 적습니다. "
          "찾아봤지만 없으면 `없음`이라고 적습니다. 수행준거의 첫 후보 줄에만 적으면 됩니다."),
    ("5", "번호 순서대로 진행합니다. 24개마다 대분류를 한 바퀴 돌므로 어디서 멈춰도 됩니다. `시범`인 앞 24개로 기준을 먼저 점검합니다."),
    ("6", "후보가 모두 채워진 수행준거만 평가에 씁니다."),
    ("", ""),
    ("직업 기준", "Task 문장만 보지 않고 그 Task가 속한 직업(I열)까지 봅니다. 활동이 같아도 직업이 수행준거의 분야(세분류)와 "
              "다르면 `관련`으로 판정합니다. 일치·좁음·넓음·근접은 직업도 맞을 때만 씁니다."),
    ("", ""),
    ("관계유형", "판정 기준"),
    ("일치", "수행준거와 Task가 같은 범위의 일을 가리키고 직업도 맞는다. 수행준거 하나에 최대 하나만 고릅니다."),
    ("좁음", "수행준거가 Task의 일부만 담는다. 직업이 맞는다."),
    ("넓음", "수행준거가 Task 전체와 그 이상을 담는다. 직업이 맞는다."),
    ("근접", "포함 관계는 아니지만 범위가 많이 겹친다. 직업이 맞는다."),
    ("관련", "범위가 다르지만 같은 분야여서 제안이 납득된다. 또는 활동은 같지만 직업이 다르다. (최종 연계표에서는 제외)"),
    ("오류", "관련이 없다. (최종 연계표에서는 제외)"),
]


def embed_unique(model, texts: list[str], prefix: str) -> np.ndarray:
    """서로 다른 문장만 길이 순으로 계산해 원래 순서로 펼친다."""
    unique = sorted(set(texts), key=lambda t: (len(t), t))
    vecs = model.encode([prefix + t for t in unique], batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    position = {t: i for i, t in enumerate(unique)}
    return vecs[[position[t] for t in texts]]


def standard(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() or 1.0)


def write_book(path, sample, pools, ncs, onet, tasks_sheet: bool) -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = "검증"
    sheet.append(["번호", "구분", "대분류", "세분류", "능력단위", "능력단위요소", "수행준거", "후보",
                  "O*NET 직업", "O*NET Task", "관계유형", "메모", "더 맞는 Task ID", "수행준거ID", "Task ID"])
    shade = PatternFill("solid", start_color="F2F2F2")
    for number, i in enumerate(sample, 1):
        r = ncs[i]
        for cand, j in enumerate(pools[number - 1], 1):
            o = onet[j]
            sheet.append([number, "시범" if number <= 24 else "본", r["대분류코드명"], r["세분류코드명"],
                          r["능력단위명칭"], r["능력단위요소명"], r["수행준거"], cand, o["Title"], o["Task"],
                          None, None, None, r["수행준거ID"], o["Task ID"]])
            if number % 2 == 0:
                for cell in sheet[sheet.max_row]:
                    cell.fill = shade
    for column, width in zip("ABCDEFGHIJKLMNO", [6, 6, 14, 18, 24, 24, 48, 6, 28, 60, 10, 24, 14, 22, 9]):
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
    guide.column_dimensions["B"].width = 110
    for row in guide.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if row[0].value in ("검증 방법", "직업 기준", "관계유형"):
            for cell in row:
                cell.font = Font(bold=True)
    if tasks_sheet:
        listing = book.create_sheet("Task목록")
        listing.append(["Task ID", "O*NET 직업", "O*NET Task"])
        for o in onet:
            listing.append([o["Task ID"], o["Title"], o["Task"]])
        for column, width in zip("ABC", [9, 36, 110]):
            listing.column_dimensions[column].width = width
        listing.freeze_panes = "A2"
        listing.auto_filter.ref = listing.dimensions
    book.save(path)


def main() -> int:
    xlsx = VALID / "validation2.xlsx"
    if xlsx.exists():
        print(f"{xlsx.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다.")
        return 1
    for d in (EMB3, WORK):
        d.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()
    n_task = len(onet)

    # 1. 표본: 대분류마다 20개, 1번 묶음에 쓴 것은 제외. 번호는 대분류를 돌아가며 매긴다.
    with (OUT / "validation" / "batch1_key.csv").open(encoding="utf-8-sig", newline="") as f:
        used = {r["수행준거ID"] for r in csv.DictReader(f)}
    rng = random.Random(SEED)
    by_major = defaultdict(list)
    for i, r in enumerate(ncs):
        if r["수행준거ID"] not in used:
            by_major[r["대분류코드"]].append(i)
    picked = [rng.sample(rows, PER_MAJOR) for _, rows in sorted(by_major.items())]
    sample = [rows[turn] for turn in range(PER_MAJOR) for rows in picked]
    ref = sorted(rng.sample(range(len(ncs)), REF))
    S = np.array(sample)

    # 2. e5-instruct 임베딩 (Task 전체, 표본 수행준거, 쏠림 보정용 수행준거 표본)
    jobs = {"onet_A": (build_texts("onet_A", ncs, onet), ""), "onet_B": (build_texts("onet_B", ncs, onet), "")}
    for style in "AB":
        texts = build_texts(f"ncs_{style}", ncs, onet)
        jobs[f"sample_{style}"] = ([texts[i] for i in sample], QUERY)
        jobs[f"ref_{style}"] = ([texts[i] for i in ref], QUERY)
    model = None
    for name, (texts, prefix) in jobs.items():
        if (EMB3 / f"{name}.npy").exists() and len(np.load(EMB3 / f"{name}.npy", mmap_mode="r")) == len(texts):
            continue
        if model is None:
            model = SentenceTransformer(INSTRUCT, device=pick_device())
            model.max_seq_length = 512
        started = time.time()
        np.save(EMB3 / f"{name}.npy", embed_unique(model, texts, prefix))
        print(f"e5-instruct {name}: {len(texts):,}문장 ({time.time() - started:.0f}초)", flush=True)
    if model is not None:
        del model
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    inst = {name: np.load(EMB3 / f"{name}.npy").astype(np.float32) for name in jobs}

    # 3. 표본 수행준거 × 전체 Task 점수
    e5 = {k: np.load(EMB / f"{k}.npy", mmap_mode="r") for k in ("ncs_A", "ncs_B", "onet_A", "onet_B")}
    order = np.argsort(S)
    back = np.argsort(order)
    sim_a = (e5["ncs_A"][S[order]].astype(np.float32) @ e5["onet_A"][:].astype(np.float32).T)[back]
    sim_b = (e5["ncs_B"][S[order]].astype(np.float32) @ e5["onet_B"][:].astype(np.float32).T)[back]
    near_e5 = np.load(OUT / "candidates" / "M3.npz")["rev_score"].mean(axis=1)
    score = {"M1": sim_a, "M3": 0.5 * sim_a + 0.5 * sim_b}
    score["M5"] = 2 * score["M3"] - near_e5[None, :]   # 수행준거 쪽 항은 순위에 영향이 없어 뺀다
    inst_mix = 0.5 * inst["sample_A"] @ inst["onet_A"].T + 0.5 * inst["sample_B"] @ inst["onet_B"].T
    best = np.zeros((n_task, 0), np.float32)
    for start in range(0, REF, 2000):
        part = 0.5 * inst["ref_A"][start:start + 2000] @ inst["onet_A"].T \
            + 0.5 * inst["ref_B"][start:start + 2000] @ inst["onet_B"].T
        best = -np.sort(-np.concatenate([best, -np.sort(-part.T, axis=1)[:, :K]], axis=1), axis=1)[:, :K]
    score["IN"] = 2 * inst_mix - best.mean(axis=1)[None, :]

    def top(values, k):
        part = np.argpartition(-values, k - 1, axis=1)[:, :k]
        return np.take_along_axis(part, np.argsort(-np.take_along_axis(values, part, axis=1), axis=1), axis=1)

    short = [sorted(set(a.tolist()) | set(b.tolist())) for a, b in zip(top(score["M5"], SHORT), top(score["IN"], SHORT))]
    pairs = [(n, j) for n, js in enumerate(short) for j in js]

    # 4. 재순위 (문장만 / 상위 분류 이름 포함)
    cache = WORK / "rerank.npz"
    if cache.exists() and np.array_equal(np.load(cache)["pairs"], np.array(pairs)):
        rerank = {s: np.load(cache)[s] for s in "AB"}
    else:
        text = {s: (build_texts(f"ncs_{s}", ncs, onet), build_texts(f"onet_{s}", ncs, onet)) for s in "AB"}
        cross = CrossEncoder(RERANKER, device=pick_device(), max_length=256)
        rerank = {}
        for s in "AB":
            started = time.time()
            rerank[s] = np.asarray(cross.predict([(text[s][0][sample[n]], text[s][1][j]) for n, j in pairs],
                                                 batch_size=16), np.float64)
            print(f"재순위 {s}: {len(pairs):,}쌍 ({time.time() - started:.0f}초)", flush=True)
        np.savez(cache, pairs=np.array(pairs), A=rerank["A"], B=rerank["B"])
        del cross
    rr_all = standard(rerank["A"]) + standard(rerank["B"])

    # 5. 후보: 방법마다 상위 5개
    rr, en = {}, {}
    at = defaultdict(list)
    for p, (n, j) in enumerate(pairs):
        at[n].append(p)
    for n, ps in at.items():
        js = [pairs[p][1] for p in ps]
        r = rr_all[ps]
        e = standard(score["M5"][n, js]) + standard(score["IN"][n, js]) + standard(r)
        rr[n] = dict(zip(js, r.tolist()))
        en[n] = dict(zip(js, e.tolist()))
    full_top = {m: top(score[m], POOL) for m in ("M5", "IN")}
    pools, key_rows = [], []
    for n in range(len(sample)):
        by_rr = sorted(rr[n], key=lambda j: -rr[n][j])
        by_en = sorted(en[n], key=lambda j: -en[n][j])
        pool = sorted(set(full_top["M5"][n].tolist()) | set(full_top["IN"][n].tolist())
                      | set(by_rr[:POOL]) | set(by_en[:POOL]))
        rng.shuffle(pool)
        pools.append(pool)
        for cand, j in enumerate(pool, 1):
            row = {"번호": n + 1, "후보": cand, "수행준거ID": ncs[sample[n]]["수행준거ID"], "Task ID": onet[j]["Task ID"]}
            for m in ("M1", "M3", "M5", "IN"):   # 전체 Task 안에서의 순위
                row[f"{m}_순위"] = int((score[m][n] > score[m][n, j]).sum()) + 1
                row[f"{m}_점수"] = f"{score[m][n, j]:.4f}"
            for m, values, ranking in (("RR", rr[n], by_rr), ("EN", en[n], by_en)):   # 재순위 후보 안에서의 순위
                row[f"{m}_순위"] = ranking.index(j) + 1 if j in values else ""
                row[f"{m}_점수"] = f"{values[j]:.4f}" if j in values else ""
            key_rows.append(row)

    write_book(xlsx, sample, pools, ncs, onet, tasks_sheet=True)
    write_book(VALID / "validation2_검증자2.xlsx", sample[:RATER2], pools[:RATER2], ncs, onet, tasks_sheet=True)
    with (VALID / "validation2_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    sizes = [len(p) for p in pools]
    print(f"수행준거 {len(sample)}개, 판정할 쌍 {len(key_rows):,}개 (수행준거당 평균 {np.mean(sizes):.1f}개, "
          f"최소 {min(sizes)}, 최대 {max(sizes)})")
    print(f"시범(앞 24개) {sum(sizes[:24])}쌍, 두 번째 검증자용(앞 {RATER2}개) {sum(sizes[:RATER2])}쌍")
    names = ["M5", "IN", "RR", "EN"]
    tops = {"M5": full_top["M5"], "IN": full_top["IN"],
            "RR": [sorted(rr[n], key=lambda j: -rr[n][j])[:POOL] for n in range(len(sample))],
            "EN": [sorted(en[n], key=lambda j: -en[n][j])[:POOL] for n in range(len(sample))]}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = np.mean([len(set(map(int, tops[a][n])) & set(map(int, tops[b][n]))) for n in range(len(sample))])
            print(f"  {a}-{b} 상위 5개 중 겹치는 Task 평균 {same:.1f}개")
    print(f"저장: {xlsx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
