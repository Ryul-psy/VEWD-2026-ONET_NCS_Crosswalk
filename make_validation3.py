"""세 번째 수행준거 검증표를 만든다 (최종 모델 후보 비교용).

    python make_validation3.py

2번 묶음에서 e5-instruct(IN)가 기본 검색 모델로 정해졌다. 여기서는 IN의 상위 50개 Task를
다시 정렬하는 방법들을 새 수행준거로 비교한다. 가중치는 판정을 보기 전에 모두 1로 고정한다.

    IN  multilingual-e5-large-instruct, 0.5 × 방식 A + 0.5 × 방식 B, 쏠림 보정 (2번 묶음과 같음)
    OC  IN + 직업 가산점       z(IN) + z(세분류→직업 투표 점수)
    RB  재순위 모델 B형식만    bge-reranker-v2-m3, 상위 분류 이름을 붙인 문장
    IB  IN + 재순위            z(IN) + z(RB)
    FN  셋 모두                z(IN) + z(직업) + z(RB)

z는 그 수행준거의 IN 상위 50개 안에서 표준화한 값이다. 직업 점수는 make_occ_validation.py의
투표 방식과 같다 (세분류의 수행준거들이 찾은 M3 상위 10개 Task의 직업을 세고 흔한 직업은 깎음).

표본: 대분류 24개 × 10개 = 수행준거 240개. 1·2번 묶음에 쓴 수행준거는 뺀다.
후보: 다섯 방법의 상위 5개를 합쳐 무작위 순서로 보여 준다.

결과  data-ONET-NCS/validation3/
        validation3.xlsx            검증자가 `관계유형` 열을 채우는 표
        validation3_검증자2.xlsx    두 번째 검증자용 (앞 72개 수행준거)
        validation3_key.csv         후보별 방법 순위와 점수
        work/short.npz              수행준거별 IN 상위 50개의 성분 점수 (다른 가중치를 나중에 시험할 때)
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import defaultdict

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from e5_embedder import pick_device
from linkdata import OUT, ROOT, build_texts, read_ncs, read_onet
from make_validation2 import EMB3, INSTRUCT, QUERY, RERANKER, embed_unique, standard, write_book

PER_MAJOR = 10
POOL = 5          # 방법마다 후보로 올리는 상위 Task 수
SHORT = 50        # 다시 정렬하는 IN 상위 Task 수
K = 10
RATER2 = 72
SEED = 20261118
METHODS = ["IN", "OC", "RB", "IB", "FN"]
VALID = ROOT / "data-ONET-NCS" / "validation3"
WORK = VALID / "work"


def main() -> int:
    xlsx = VALID / "validation3.xlsx"
    if xlsx.exists():
        print(f"{xlsx.name}이 이미 있습니다. 판정을 덮어쓰지 않도록 멈춥니다.")
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()

    # 1. 표본: 대분류마다 10개, 앞선 두 묶음에 쓴 것은 제외
    used = set()
    for path in (OUT / "validation" / "batch1_key.csv", ROOT / "data-ONET-NCS" / "validation2" / "validation2_key.csv"):
        with path.open(encoding="utf-8-sig", newline="") as f:
            used |= {r["수행준거ID"] for r in csv.DictReader(f)}
    rng = random.Random(SEED)
    by_major = defaultdict(list)
    for i, r in enumerate(ncs):
        if r["수행준거ID"] not in used:
            by_major[r["대분류코드"]].append(i)
    picked = [rng.sample(rows, PER_MAJOR) for _, rows in sorted(by_major.items())]
    sample = [rows[turn] for turn in range(PER_MAJOR) for rows in picked]

    # 2. e5-instruct 점수 (Task·보정용 표본 임베딩은 2번 묶음 것을 그대로 쓴다)
    texts = {s: build_texts(f"ncs_{s}", ncs, onet) for s in "AB"}
    task_text = build_texts("onet_B", ncs, onet)
    inst = {name: np.load(EMB3 / f"{name}.npy").astype(np.float32) for name in ("onet_A", "onet_B", "ref_A", "ref_B")}
    model = None
    for s in "AB":
        path = EMB3 / f"sample3_{s}.npy"
        if not (path.exists() and len(np.load(path, mmap_mode="r")) == len(sample)):
            if model is None:
                model = SentenceTransformer(INSTRUCT, device=pick_device())
                model.max_seq_length = 512
            np.save(path, embed_unique(model, [texts[s][i] for i in sample], QUERY))
        inst[f"sample_{s}"] = np.load(path).astype(np.float32)
    del model
    best = np.zeros((len(onet), 0), np.float32)
    for start in range(0, len(inst["ref_A"]), 2000):
        part = 0.5 * inst["ref_A"][start:start + 2000] @ inst["onet_A"].T \
            + 0.5 * inst["ref_B"][start:start + 2000] @ inst["onet_B"].T
        best = -np.sort(-np.concatenate([best, -np.sort(-part.T, axis=1)[:, :K]], axis=1), axis=1)[:, :K]
    score_in = 2 * (0.5 * inst["sample_A"] @ inst["onet_A"].T + 0.5 * inst["sample_B"] @ inst["onet_B"].T) \
        - best.mean(axis=1)[None, :]
    short = np.argsort(-score_in, axis=1, kind="stable")[:, :SHORT]      # IN 순위 순서

    # 3. 직업 투표 점수 (make_occ_validation.py와 같은 계산)
    occs = sorted({r["O*NET-SOC Code"] for r in onet})
    subs = sorted({r["NCS세분류코드"] for r in ncs})
    opos, spos = {c: k for k, c in enumerate(occs)}, {c: k for k, c in enumerate(subs)}
    task_occ = np.array([opos[r["O*NET-SOC Code"]] for r in onet])
    crit_sub = np.array([spos[r["NCS세분류코드"]] for r in ncs])
    top = np.load(OUT / "candidates" / "M3.npz")["fwd_idx"]
    votes = np.zeros((len(subs), len(occs)), np.float64)
    np.add.at(votes, (np.repeat(crit_sub, top.shape[1]), task_occ[top].ravel()), 1)
    lift = (votes / votes.sum(1, keepdims=True)) / (votes.sum(0, keepdims=True) / votes.sum())
    vote = votes * np.log(np.maximum(lift, 1.0))
    occ = np.array([vote[crit_sub[i], task_occ[short[n]]] for n, i in enumerate(sample)])

    # 4. 재순위 (B형식: 상위 분류 이름 포함)
    cache = WORK / "rerank.npz"
    if cache.exists() and np.array_equal(np.load(cache)["short"], short):
        prob = np.load(cache)["B"]
    else:
        cross = CrossEncoder(RERANKER, device=pick_device(), max_length=256)
        started = time.time()
        prob = np.asarray(cross.predict([(texts["B"][i], task_text[j]) for n, i in enumerate(sample) for j in short[n]],
                                        batch_size=16), np.float64).reshape(short.shape)
        print(f"재순위: {short.size:,}쌍 ({time.time() - started:.0f}초)", flush=True)
        np.savez(cache, short=short, B=prob)
        del cross
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    logit = np.log(clipped / (1 - clipped))

    # 5. 방법별 점수와 후보
    base = np.take_along_axis(score_in, short, axis=1)
    z = {"IN": np.array([standard(r) for r in base]), "직업": np.array([standard(r) for r in occ]),
         "RB": np.array([standard(r) for r in logit])}
    score = {"IN": base, "OC": z["IN"] + z["직업"], "RB": logit, "IB": z["IN"] + z["RB"],
             "FN": z["IN"] + z["직업"] + z["RB"]}
    order = {m: np.argsort(-s, axis=1, kind="stable") for m, s in score.items()}   # 동점이면 IN 순위가 앞
    rank = {m: np.argsort(o, axis=1) + 1 for m, o in order.items()}
    np.savez(WORK / "short.npz", sample=np.array(sample), short=short, IN=base, 직업=occ, RB_확률=prob, RB=logit)

    pools, key_rows = [], []
    for n, i in enumerate(sample):
        places = sorted({int(p) for m in METHODS for p in order[m][n, :POOL]})
        rng.shuffle(places)
        pools.append([int(short[n, p]) for p in places])
        for cand, p in enumerate(places, 1):
            row = {"번호": n + 1, "후보": cand, "수행준거ID": ncs[i]["수행준거ID"], "Task ID": onet[short[n, p]]["Task ID"]}
            for m in METHODS:
                row[f"{m}_순위"] = int(rank[m][n, p])
                row[f"{m}_점수"] = f"{score[m][n, p]:.6f}"
            row["직업_점수"] = f"{occ[n, p]:.6f}"
            row["직업코드"] = onet[short[n, p]]["O*NET-SOC Code"]
            key_rows.append(row)

    write_book(xlsx, sample, pools, ncs, onet, tasks_sheet=True)
    write_book(VALID / "validation3_검증자2.xlsx", sample[:RATER2], pools[:RATER2], ncs, onet, tasks_sheet=True)
    with (VALID / "validation3_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    sizes = [len(p) for p in pools]
    print(f"수행준거 {len(sample)}개, 판정할 쌍 {len(key_rows):,}개 (수행준거당 평균 {np.mean(sizes):.1f}개, "
          f"최소 {min(sizes)}, 최대 {max(sizes)})")
    print(f"시범(앞 24개) {sum(sizes[:24])}쌍, 두 번째 검증자용(앞 {RATER2}개) {sum(sizes[:RATER2])}쌍")
    for a_i, a in enumerate(METHODS):
        for b in METHODS[a_i + 1:]:
            same = np.mean([len(set(order[a][n, :POOL]) & set(order[b][n, :POOL])) for n in range(len(sample))])
            first = np.mean(order[a][:, 0] == order[b][:, 0])
            print(f"  {a}-{b}: 상위 5개 중 겹치는 Task 평균 {same:.1f}개, 1위가 같은 수행준거 {first:.0%}")
    print(f"저장: {xlsx.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
