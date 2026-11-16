"""onet_B.npy를 고친 뒤 다시 계산한 모델을 1번 묶음 판정으로 비교한다.

    python evaluate_recalc.py

검증표(batch1_validation.xlsx)와 batch1_key.csv는 읽기만 한다. key의 M3 순위는 손상된
방식 B 벡터로 계산한 옛 값이므로, 여기서는 고친 임베딩과 새 후보(candidates/*.npz)로 다시 매긴다.

두 가지로 본다.
    1. 판정된 후보 안에서 다시 정렬 (모든 모델이 같은 후보를 놓고 겨룬다)
    2. 전체 Task 18,796개에서 뽑은 새 상위 k (판정되지 않은 Task는 틀린 것으로 센 하한)
동점은 불리한 쪽으로 처리한다 (맞는 연결이 뒤).

결과: data-ONET-NCS/batch1/validation/batch1_evaluation_recalc.md
      data-ONET-NCS/batch1/validation/batch1_key_recalc.csv  (쌍마다 새 점수와 후보 안 순위)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import comb

import numpy as np
from openpyxl import load_workbook

from find_candidates import ALL_MODELS, CAND, CSLS, MODELS
from linkdata import EMB, OUT, read_ncs, read_onet

GOOD = {"일치", "좁음", "넓음", "근접"}
KS = [1, 2, 3, 5]
VALID = OUT / "validation"


def sign_p(win: int, lose: int) -> float:
    """양측 부호검정."""
    n, k = win + lose, min(win, lose)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    sheet = load_workbook(VALID / "batch1_validation.xlsx", read_only=True)["검증"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    col = {name: header.index(name) for name in ("번호", "후보", "관계유형")}
    label = {(int(r[col["번호"]]), int(r[col["후보"]])): r[col["관계유형"]] for r in rows if r[0] is not None}
    with (VALID / "batch1_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = [r for r in csv.DictReader(f) if label.get((int(r["번호"]), int(r["후보"])))]
    good = np.array([label[(int(r["번호"]), int(r["후보"]))] in GOOD for r in key])

    ncs_row = {r["수행준거ID"]: i for i, r in enumerate(read_ncs())}
    task_row = {r["Task ID"]: i for i, r in enumerate(read_onet())}
    ci = np.array([ncs_row[r["수행준거ID"]] for r in key])
    ti = np.array([task_row[r["Task ID"]] for r in key])
    by_item = defaultdict(list)
    for i, r in enumerate(key):
        by_item[int(r["번호"])].append(i)
    matched = [n for n, ix in by_item.items() if good[ix].any()]

    # 쌍 점수: 고친 임베딩에서 직접 계산
    emb = {k: np.load(EMB / f"{k}.npy", mmap_mode="r") for k in ("ncs_A", "ncs_B", "onet_A", "onet_B")}
    crit = sorted(set(ci.tolist()))
    pos = {c: i for i, c in enumerate(crit)}
    cos = {}
    for s in "AB":
        q = emb[f"ncs_{s}"][crit].astype(np.float32)
        t = emb[f"onet_{s}"][np.sort(np.unique(ti))].astype(np.float32)
        tpos = {j: i for i, j in enumerate(np.sort(np.unique(ti)).tolist())}
        cos[s] = np.array([q[pos[c]] @ t[tpos[j]] for c, j in zip(ci.tolist(), ti.tolist())])
    npz = {m: np.load(CAND / f"{m}.npz") for m in ALL_MODELS}
    score = {m: w_a * cos["A"] + w_b * cos["B"] for m, (w_a, w_b) in MODELS.items()}
    for m, base in CSLS.items():
        near_ncs = npz[base]["fwd_score"].mean(axis=1)
        near_task = npz[base]["rev_score"].mean(axis=1)
        score[m] = 2 * score[base] - near_ncs[ci] - near_task[ti]

    def pool_rank(values):
        rank = np.zeros(len(key), int)
        for ix in by_item.values():
            for place, i in enumerate(sorted(ix, key=lambda i: (-values[i], good[i])), 1):
                rank[i] = place
        return rank

    rank = {m: pool_rank(score[m]) for m in ALL_MODELS}
    old = {m: np.array([int(r[f"{m}_순위"]) if r.get(f"{m}_순위") else 99 for r in key]) for m in ("M1", "M3", "M4")}

    def hits(r, k):
        return np.array([bool((good[by_item[n]] & (r[by_item[n]] <= k)).any()) for n in matched])

    base = hits(old["M3"], 1)
    out = ["# 1번 묶음 모델 비교: onet_B.npy 수정 후 재계산", "",
           f"- 판정된 쌍 {len(key):,}개, 수행준거 {len(by_item)}개, 맞는 연결이 있는 수행준거 {len(matched)}개",
           "- 동점은 불리한 쪽으로 처리. `옛 M3 대비`는 상위 1에서 이긴 수행준거 / 진 수행준거 (양측 부호검정)", "",
           "## 1. 판정된 후보 안에서 다시 정렬", "",
           "| 모델 | " + " | ".join(f"상위 {k}" for k in KS) + " | 1위가 `오류` | 옛 M3 대비 |",
           "| --- |" + " --- |" * (len(KS) + 2)]
    lab = np.array([label[(int(r["번호"]), int(r["후보"]))] for r in key])
    table = [(f"옛 {m} (수정 전, 원래 순위)", old[m]) for m in ("M1", "M3", "M4")] + [(m, rank[m]) for m in ALL_MODELS]
    for name, r in table:
        h = hits(r, 1)
        win, lose = int((h & ~base).sum()), int((~h & base).sum())
        wrong = sum(bool(((r[ix] == 1) & (lab[ix] == "오류")).any()) for ix in by_item.values())
        versus = "기준" if name.startswith("옛 M3") else f"+{win}/-{lose} (p={sign_p(win, lose):.2f})"
        out.append(f"| {name} | " + " | ".join(f"{hits(r, k).mean():.1%}" for k in KS) + f" | {wrong} | {versus} |")

    # 전체 Task에서 새로 뽑은 상위 k
    out += ["", "## 2. 전체 Task에서 뽑은 새 상위 k (판정되지 않은 Task는 틀린 것으로 센 하한)", "",
            "| 모델 | " + " | ".join(f"상위 {k}" for k in KS) + " | 새 1위가 판정된 비율 | 새 상위 5개가 판정된 비율 |",
            "| --- |" + " --- |" * (len(KS) + 2)]
    rated = defaultdict(dict)
    for i in range(len(key)):
        rated[int(ci[i])][int(ti[i])] = bool(good[i])
    item_crit = {n: int(ci[ix[0]]) for n, ix in by_item.items()}
    for m in ALL_MODELS:
        top = npz[m]["fwd_idx"]
        acc = [np.mean([any(rated[item_crit[n]].get(int(j), False) for j in top[item_crit[n], :k]) for n in matched])
               for k in KS]
        r1 = np.mean([int(top[c, 0]) in rated[c] for c in item_crit.values()])
        r5 = np.mean([int(j) in rated[c] for c in item_crit.values() for j in top[c, :5]])
        out.append(f"| {m} | " + " | ".join(f"{a:.1%}" for a in acc) + f" | {r1:.1%} | {r5:.1%} |")
    out.append("")

    with (VALID / "batch1_key_recalc.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["번호", "후보", "수행준거ID", "Task ID"] + [f"{m}_{c}" for m in ALL_MODELS for c in ("후보안순위", "점수")])
        for i, r in enumerate(key):
            writer.writerow([r["번호"], r["후보"], r["수행준거ID"], r["Task ID"]]
                            + [x for m in ALL_MODELS for x in (rank[m][i], f"{score[m][i]:.4f}")])
    text = "\n".join(out)
    (VALID / "batch1_evaluation_recalc.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
