"""재순위 모델 시험: batch1(multilingual-e5-large)에서 사람이 판정한 후보를 다시 정렬해 본다.

    python rerank_test.py                                  # BAAI/bge-reranker-v2-m3
    python rerank_test.py --model dragonkue/bge-reranker-v2-m3-ko

재순위 모델은 수행준거와 Task를 한 쌍으로 함께 읽고 점수를 낸다. 판정된 쌍만 다시 정렬하므로
새 판정 없이 평가할 수 있다. 수행준거마다 후보는 batch1의 M1·M3·M4 상위 5개를 합친 것이다.

결과: data-ONET-NCS/batch2/
    scores_{모델}.csv    쌍마다 재순위 점수 (문장만 / 상위 분류 이름 포함)
    evaluation.md        batch1 모델과의 비교 (모델을 바꿔 실행하면 뒤에 덧붙는다)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict

import numpy as np
from openpyxl import load_workbook
from sentence_transformers import CrossEncoder

from e5_embedder import pick_device
from linkdata import OUT, ROOT, read_ncs, read_onet

OUT2 = ROOT / "data-ONET-NCS" / "batch2"
GOOD = {"일치", "좁음", "넓음", "근접"}
KS = [1, 2, 3, 5]


def auc(pos: list[float], neg: list[float]) -> float:
    """pos에서 하나, neg에서 하나를 뽑았을 때 pos 쪽 점수가 더 높을 확률."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + (pos[:, None] == neg[None, :]).mean() / 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    OUT2.mkdir(parents=True, exist_ok=True)

    ncs = {r["수행준거ID"]: r for r in read_ncs()}
    onet = {r["Task ID"]: r for r in read_onet()}
    sheet = load_workbook(OUT / "validation" / "batch1_validation.xlsx", read_only=True)["검증"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    col = {name: header.index(name) for name in ("번호", "후보", "관계유형")}
    label = {(int(r[col["번호"]]), int(r[col["후보"]])): r[col["관계유형"]] for r in rows if r[0] is not None}
    with (OUT / "validation" / "batch1_key.csv").open(encoding="utf-8-sig", newline="") as f:
        key = [r for r in csv.DictReader(f) if label.get((int(r["번호"]), int(r["후보"])))]
    for r in key:
        r["관계유형"] = label[(int(r["번호"]), int(r["후보"]))]

    # 쌍 문장: A = 문장만, B = 상위 분류 이름 포함 (batch1의 방식 A·B와 같다)
    pairs = {"A": [], "B": []}
    for r in key:
        n, o = ncs[r["수행준거ID"]], onet[r["Task ID"]]
        pairs["A"].append((n["수행준거"], o["Task"]))
        pairs["B"].append((f'{n["세분류코드명"]} > {n["능력단위명칭"]} > {n["능력단위요소명"]}: {n["수행준거"]}',
                           f'{o["Title"]}: {o["Task"]}'))
    model = CrossEncoder(args.model, device=pick_device(), max_length=256)
    started = time.time()
    score = {style: np.asarray(model.predict(p, batch_size=args.batch_size), np.float64) for style, p in pairs.items()}
    rate = 2 * len(key) / (time.time() - started)

    def standard(x):
        return (x - x.mean()) / x.std()

    # 비교할 순서: batch1 모델은 원래 순위, 재순위는 후보 안에서 점수 순
    e5 = np.array([float(r["M3_점수"] or "nan") for r in key])
    methods = {f"재순위 A (문장만)": score["A"], "재순위 B (상위 분류 포함)": score["B"],
               "재순위 A+B 평균": standard(score["A"]) + standard(score["B"])}
    by_item = defaultdict(list)
    for i, r in enumerate(key):
        by_item[int(r["번호"])].append(i)
    matched = {n: ix for n, ix in by_item.items() if any(key[i]["관계유형"] in GOOD for i in ix)}

    def rank_in_pool(values):
        rank = np.zeros(len(key), int)
        for ix in by_item.values():
            for place, i in enumerate(sorted(ix, key=lambda i: -values[i]), 1):
                rank[i] = place
        return rank

    ranks = {m: np.array([int(r[f"{m}_순위"]) if r[f"{m}_순위"] else 99 for r in key]) for m in ("M1", "M3", "M4")}
    ranks.update({name: rank_in_pool(v) for name, v in methods.items()})
    rng = np.random.default_rng(0)
    random_top1 = np.mean([np.mean([key[i]["관계유형"] in GOOD for i in ix]) for ix in matched.values()])

    def hit(name, ix, k):
        return any(ranks[name][i] <= k and key[i]["관계유형"] in GOOD for i in ix)

    short = args.model.split("/")[-1]
    out = [f"## {args.model}", "",
           f"- 판정된 쌍 {len(key):,}개, 수행준거 {len(by_item)}개 (맞는 연결이 있는 수행준거 {len(matched)}개)",
           f"- 계산 속도: 초당 {rate:.0f}쌍 ({pick_device()})",
           f"- 후보 가운데 무작위로 하나를 고를 때의 상위 1 적중률: {random_top1:.1%}", "",
           "### 상위 k 적중률 (맞는 연결이 있는 수행준거 기준)", "",
           "| 방법 | " + " | ".join(f"상위 {k}" for k in KS) + " | 1위가 `오류` |", "| --- |" + " --- |" * (len(KS) + 1)]
    for name in ranks:
        wrong = sum(any(ranks[name][i] == 1 and key[i]["관계유형"] == "오류" for i in ix) for ix in by_item.values())
        label_name = f"batch1 {name}" if name in ("M1", "M3", "M4") else name
        out.append(f"| {label_name} | " + " | ".join(
            f"{np.mean([hit(name, ix, k) for ix in matched.values()]):.1%}" for k in KS) + f" | {wrong} |")

    # 쌍 단위 구분력과, 대응 Task가 없는 수행준거를 가려내는 힘
    good = np.array([r["관계유형"] in GOOD for r in key])
    out += ["", "### 점수의 구분력 (0.5 = 구분 못 함, 1 = 완전히 구분)", "",
            "| 점수 | 맞는 쌍 대 나머지 쌍 | 맞는 연결이 있는 수행준거 대 없는 수행준거 (최고 점수 기준) |", "| --- | --- | --- |"]
    for name, v in {"batch1 M3 점수 (상위 10위 안 쌍만)": e5, **methods}.items():
        ok = ~np.isnan(v)
        best = {n: np.nanmax(v[ix]) for n, ix in by_item.items() if not np.all(np.isnan(v[ix]))}
        out.append(f"| {name} | {auc(v[ok & good], v[ok & ~good]):.3f} | "
                   f"{auc([b for n, b in best.items() if n in matched], [b for n, b in best.items() if n not in matched]):.3f} |")
    out.append("")

    with (OUT2 / f"scores_{short}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["번호", "후보", "수행준거ID", "Task ID", "관계유형", "재순위_A", "재순위_B"])
        for i, r in enumerate(key):
            writer.writerow([r["번호"], r["후보"], r["수행준거ID"], r["Task ID"], r["관계유형"],
                             f'{score["A"][i]:.4f}', f'{score["B"][i]:.4f}'])
    report = OUT2 / "evaluation.md"
    head = "" if report.exists() else "# 재순위 모델 시험 (batch1 판정 자료 기준)\n\n"
    with report.open("a", encoding="utf-8") as f:
        f.write(head + "\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
