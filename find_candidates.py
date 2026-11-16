"""모델 3가지로 수행준거마다 Task 상위 10개를 찾는다 (METHOD.md 4~5절).

    python find_candidates.py

모델 (모두 multilingual-e5-large 임베딩, 점수 계산 방식만 다름)
    M1  방식 A 유사도 (문장만)
    M3  0.5 × 방식 A + 0.5 × 방식 B
    (M2 = 방식 B만 쓰는 모델은 무관한 후보가 많아 뺐다. 번호는 그대로 둔다.)
    M4  방식 A 유사도에서 쏠림을 보정한 점수 (CSLS)
        = 2 × 유사도 − 그 수행준거의 상위 10개 평균 − 그 Task의 상위 10개 평균
        아무 수행준거에나 높게 나오는 Task의 점수를 깎는다.
    M5  M3 점수에 같은 쏠림 보정을 한 점수

결과: data-ONET-NCS/batch1/candidates/
    M1.csv, M3.csv, M4.csv, M5.csv   수행준거마다 Task 상위 10개와 점수
    reverse_M1.csv ...            Task마다 수행준거 상위 10개와 점수 (반대 방향)
    M1.npz ...                    같은 내용의 배열 (다른 스크립트가 읽는다)
    summary.txt                   모델별 요약
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter

import numpy as np

from linkdata import EMB, OUT, read_ncs, read_onet

K = 10
CHUNK = 2000
MODELS = {"M1": (1.0, 0.0), "M3": (0.5, 0.5)}  # (방식 A 가중치, 방식 B 가중치)
CSLS = {"M4": "M1", "M5": "M3"}  # 쏠림 보정 모델: 바탕이 되는 모델
ALL_MODELS = list(MODELS) + list(CSLS)
CAND = OUT / "candidates"


def top_k(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """행마다 점수가 높은 K개의 위치와 점수를 높은 순서로 돌려준다."""
    part = np.argpartition(-sim, K - 1, axis=1)[:, :K]
    scores = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(scores, order, axis=1)


def main() -> int:
    CAND.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()
    ncs_A = np.load(EMB / "ncs_A.npy", mmap_mode="r")
    ncs_B = np.load(EMB / "ncs_B.npy", mmap_mode="r")
    onet_A = np.load(EMB / "onet_A.npy").astype(np.float32)
    onet_B = np.load(EMB / "onet_B.npy").astype(np.float32)
    if len(ncs_A) != len(ncs) or len(onet_A) != len(onet):
        print("임베딩과 자료의 행 수가 다릅니다. embed_texts.py를 다시 실행하세요.")
        return 1
    n, t = len(ncs), len(onet)

    fwd_idx = {m: np.zeros((n, K), np.int32) for m in ALL_MODELS}
    fwd_score = {m: np.zeros((n, K), np.float32) for m in ALL_MODELS}
    rev_idx = {m: np.zeros((t, 0), np.int32) for m in ALL_MODELS}
    rev_score = {m: np.zeros((t, 0), np.float32) for m in ALL_MODELS}

    def keep(m, sim, start, stop):
        fwd_idx[m][start:stop], fwd_score[m][start:stop] = top_k(sim)
        # 반대 방향: 이 묶음의 상위 K개를 지금까지의 상위 K개와 합쳐 다시 고른다
        idx, score = top_k(np.ascontiguousarray(sim.T))
        all_idx = np.concatenate([rev_idx[m], idx + start], axis=1)
        all_score = np.concatenate([rev_score[m], score], axis=1)
        order = np.argsort(-all_score, axis=1)[:, :K]
        rev_idx[m] = np.take_along_axis(all_idx, order, axis=1)
        rev_score[m] = np.take_along_axis(all_score, order, axis=1)

    started = time.time()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        sim_A = ncs_A[start:stop].astype(np.float32) @ onet_A.T
        sim_B = ncs_B[start:stop].astype(np.float32) @ onet_B.T
        for m, (w_a, w_b) in MODELS.items():
            sim = sim_A if w_b == 0 else sim_B if w_a == 0 else w_a * sim_A + w_b * sim_B
            keep(m, sim, start, stop)
        if (start // CHUNK) % 10 == 0:
            print(f"{stop:,}/{n:,} ({time.time() - started:.0f}초)", flush=True)

    # 쏠림 보정 모델: 바탕 모델의 상위 K개 평균을 양쪽에서 뺀 점수로 다시 고른다
    for m, base in CSLS.items():
        w_a, w_b = MODELS[base]
        near_ncs = fwd_score[base].mean(axis=1)
        near_task = rev_score[base].mean(axis=1)
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            sim = 0
            if w_a:
                sim = sim + w_a * (ncs_A[start:stop].astype(np.float32) @ onet_A.T)
            if w_b:
                sim = sim + w_b * (ncs_B[start:stop].astype(np.float32) @ onet_B.T)
            keep(m, 2 * sim - near_ncs[start:stop, None] - near_task[None, :], start, stop)
        print(f"{m} 끝 ({time.time() - started:.0f}초)", flush=True)

    ncs_id = [r["수행준거ID"] for r in ncs]
    task_id = [r["Task ID"] for r in onet]
    lines = []
    for m in ALL_MODELS:
        np.savez_compressed(CAND / f"{m}.npz", fwd_idx=fwd_idx[m], fwd_score=fwd_score[m],
                            rev_idx=rev_idx[m], rev_score=rev_score[m])
        for name, key, ids, other, idx, score in [
            (f"{m}.csv", "수행준거ID", ncs_id, task_id, fwd_idx[m], fwd_score[m]),
            (f"reverse_{m}.csv", "Task ID", task_id, ncs_id, rev_idx[m], rev_score[m]),
        ]:
            target = "TaskID" if key == "수행준거ID" else "수행준거ID"
            with (CAND / name).open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([key] + [f"{rank}위_{c}" for rank in range(1, K + 1) for c in (target, "점수")])
                for i, row_id in enumerate(ids):
                    row = [row_id]
                    for j, s in zip(idx[i], score[i]):
                        row += [other[j], f"{s:.4f}"]
                    writer.writerow(row)

        # 요약: 점수 수준, 한 번도 후보에 오르지 않은 Task, 지나치게 자주 오르는 Task, 상호 일치
        used = Counter(fwd_idx[m].ravel().tolist())
        rev_sets = [set(row) for row in rev_idx[m].tolist()]
        sample = range(0, n, 20)
        mutual = np.mean([i in rev_sets[j] for i in sample for j in fwd_idx[m][i]])
        lines += [
            f"[{m}] " + (f"방식 A 가중치 {MODELS[m][0]}, 방식 B 가중치 {MODELS[m][1]}" if m in MODELS
                        else f"{CSLS[m]}에 쏠림 보정 (점수는 보정 점수여서 다른 모델과 견줄 수 없음)"),
            f"  1위 점수 평균 {fwd_score[m][:, 0].mean():.3f}, 10위 점수 평균 {fwd_score[m][:, -1].mean():.3f}, "
            f"1·2위 격차 평균 {(fwd_score[m][:, 0] - fwd_score[m][:, 1]).mean():.4f}",
            f"  어느 수행준거의 상위 {K}개에도 오르지 않은 Task: {t - len(used):,}개 / {t:,}개",
            f"  상위 {K}개 쌍 가운데 반대 방향에서도 상위 {K}개인 비율: {mutual:.1%}",
            "  가장 자주 후보에 오른 Task:",
        ]
        for j, count in used.most_common(5):
            lines.append(f"    {count:>6,}회  [{onet[j]['Title']}] {onet[j]['Task']}")
    pairs = [(a, b) for i, a in enumerate(ALL_MODELS) for b in ALL_MODELS[i + 1:]]
    for a, b in pairs:
        same_top1 = np.mean(fwd_idx[a][:, 0] == fwd_idx[b][:, 0])
        overlap = np.mean([len(set(x) & set(y)) for x, y in zip(fwd_idx[a][::20].tolist(), fwd_idx[b][::20].tolist())])
        lines.append(f"[{a} 대 {b}] 1위가 같은 수행준거 {same_top1:.1%}, 상위 {K}개 중 겹치는 Task 평균 {overlap:.1f}개")

    (CAND / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {CAND.relative_to(OUT.parent.parent)} ({time.time() - started:.0f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
