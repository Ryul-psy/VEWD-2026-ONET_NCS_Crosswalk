"""수행준거 전체(196,657개)에 최종 모델 FN을 적용해 수행준거 → Task 연계표를 만든다.

    caffeinate -i python build_crosswalk.py          # 전체 (며칠 걸림. 멈춰도 다시 실행하면 이어서 한다)
    python build_crosswalk.py --status                # 진행 상황만 본다

FN = z(IN) + z(직업) + z(재순위). 3번 검증표(make_validation3.py)와 같은 계산이다.
    IN      multilingual-e5-large-instruct, 0.5 × 방식 A + 0.5 × 방식 B, 쏠림 보정 (Task 쪽 항은 2번 묶음의 표본 20,000개)
    직업    세분류 → 직업 투표 점수 (make_occ_validation.py와 같음)
    재순위  bge-reranker-v2-m3, 상위 분류 이름을 붙인 문장 (B형식)
z는 수행준거마다 IN 상위 50개 Task 안에서 표준화한 값이다.

단계
    1. 수행준거 전체를 e5-instruct로 임베딩          batch3/embeddings/ncs_A.npy, ncs_B.npy
    2. 수행준거마다 IN 상위 50개 Task                crosswalk/work/in_top50.npz
    3. 직업 점수를 붙이고 재순위 없는 중간 연계표    crosswalk/ncs_onet_task_OC.csv
    4. 재순위 (수행준거 1,000개씩 저장)              crosswalk/work/rerank/part_*.npy
    5. 최종 연계표                                   crosswalk/ncs_onet_task_FN.csv, crosswalk/work/scores.npz

연계표: 수행준거마다 상위 5개 Task. 점수는 수행준거 안에서만 비교할 수 있다 (표준화 점수의 합).
"""

from __future__ import annotations

import csv
import sys
import time

import numpy as np

from linkdata import OUT, ROOT, build_texts, read_ncs, read_onet

EMB3 = ROOT / "data-ONET-NCS" / "batch3" / "embeddings"
CROSS = ROOT / "data-ONET-NCS" / "crosswalk"
WORK = CROSS / "work"
PARTS = WORK / "rerank"
INSTRUCT = "intfloat/multilingual-e5-large-instruct"
QUERY = "Instruct: Retrieve semantically similar text.\nQuery: "
RERANKER = "BAAI/bge-reranker-v2-m3"
SHORT = 50        # 다시 정렬하는 IN 상위 Task 수
TOP = 5           # 연계표에 싣는 Task 수
K = 10
EMBED_CHUNK = 20000
RERANK_CHUNK = 1000


def say(text: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M')}] {text}", flush=True)


def zrow(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(sd == 0, 1.0, sd)


def embed_all(texts: list[str], name: str) -> None:
    """서로 다른 문장만 길이 순으로, 20,000개씩 저장하며 계산한다."""
    out = EMB3 / f"{name}.npy"
    if out.exists() and len(np.load(out, mmap_mode="r")) == len(texts):
        return
    unique = sorted(set(texts), key=lambda t: (len(t), t))
    starts = list(range(0, len(unique), EMBED_CHUNK))
    part = lambda s: EMB3 / f"{name}.part{s // EMBED_CHUNK:03d}.npy"
    model = None
    for s in starts:
        chunk = unique[s:s + EMBED_CHUNK]
        if part(s).exists() and len(np.load(part(s), mmap_mode="r")) == len(chunk):
            continue
        if model is None:
            from sentence_transformers import SentenceTransformer
            from e5_embedder import pick_device
            model = SentenceTransformer(INSTRUCT, device=pick_device())
            model.max_seq_length = 512
        started = time.time()
        vecs = model.encode([QUERY + t for t in chunk], batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        np.save(part(s), vecs)
        say(f"임베딩 {name}: {min(s + EMBED_CHUNK, len(unique)):,} / {len(unique):,}문장 ({time.time() - started:.0f}초)")
    del model
    vecs = np.concatenate([np.load(part(s)) for s in starts])
    position = {t: i for i, t in enumerate(unique)}
    np.save(out, vecs[[position[t] for t in texts]])
    for s in starts:
        part(s).unlink()


def status(n: int) -> None:
    for name in ("ncs_A", "ncs_B"):
        done = (EMB3 / f"{name}.npy").exists()
        parts = len(list(EMB3.glob(f"{name}.part*.npy")))
        print(f"1. 임베딩 {name}: {'완료' if done else f'진행 중 (저장된 조각 {parts}개)'}")
    print(f"2. IN 상위 {SHORT}개: {'완료' if (WORK / 'in_top50.npz').exists() else '아직'}")
    print(f"3. 중간 연계표(OC): {'완료' if (CROSS / 'ncs_onet_task_OC.csv').exists() else '아직'}")
    total = -(-n // RERANK_CHUNK)
    print(f"4. 재순위: {len(list(PARTS.glob('part_*.npy')))} / {total} 조각")
    print(f"5. 최종 연계표(FN): {'완료' if (CROSS / 'ncs_onet_task_FN.csv').exists() else '아직'}")


def write_table(path, ncs, onet, short, order, columns: dict) -> None:
    """수행준거마다 상위 TOP개 Task를 쓴다. columns = 이름 → (수행준거, SHORT) 점수 배열."""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["수행준거ID", "NCS세분류코드", "순위", "Task ID", "O*NET-SOC Code", "O*NET 직업", "IN_순위"] + list(columns))
        for n, r in enumerate(ncs):
            for place in range(TOP):
                p = order[n, place]
                o = onet[short[n, p]]
                writer.writerow([r["수행준거ID"], r["NCS세분류코드"], place + 1, o["Task ID"], o["O*NET-SOC Code"], o["Title"],
                                 p + 1] + [f"{v[n, p]:.4f}" for v in columns.values()])


def main() -> int:
    ncs, onet = read_ncs(), read_onet()
    n = len(ncs)
    PARTS.mkdir(parents=True, exist_ok=True)
    if "--status" in sys.argv:
        status(n)
        return 0

    # 1. 임베딩
    texts = {s: build_texts(f"ncs_{s}", ncs, onet) for s in "AB"}
    for s in "AB":
        embed_all(texts[s], f"ncs_{s}")

    # 2. IN 상위 50개
    cache = WORK / "in_top50.npz"
    if not cache.exists():
        inst = {k: np.load(EMB3 / f"{k}.npy").astype(np.float32) for k in ("onet_A", "onet_B", "ref_A", "ref_B")}
        best = np.zeros((len(onet), 0), np.float32)
        for start in range(0, len(inst["ref_A"]), 2000):
            part = 0.5 * inst["ref_A"][start:start + 2000] @ inst["onet_A"].T \
                + 0.5 * inst["ref_B"][start:start + 2000] @ inst["onet_B"].T
            best = -np.sort(-np.concatenate([best, -np.sort(-part.T, axis=1)[:, :K]], axis=1), axis=1)[:, :K]
        near = best.mean(axis=1)[None, :]
        emb = {s: np.load(EMB3 / f"ncs_{s}.npy", mmap_mode="r") for s in "AB"}
        short = np.zeros((n, SHORT), np.int32)
        base = np.zeros((n, SHORT), np.float32)
        for start in range(0, n, 2000):
            rows = slice(start, start + 2000)
            score = 2 * (0.5 * emb["A"][rows].astype(np.float32) @ inst["onet_A"].T
                         + 0.5 * emb["B"][rows].astype(np.float32) @ inst["onet_B"].T) - near
            idx = np.argsort(-score, axis=1, kind="stable")[:, :SHORT]
            short[rows], base[rows] = idx, np.take_along_axis(score, idx, axis=1)
        np.savez(cache, short=short, IN=base)
        say("IN 상위 50개 계산 완료")
        del inst, emb
    short, base = np.load(cache)["short"], np.load(cache)["IN"]

    # 3. 직업 점수와 중간 연계표
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
    occ = vote[crit_sub[:, None], task_occ[short]]
    z_in, z_occ = zrow(base), zrow(occ)
    interim = CROSS / "ncs_onet_task_OC.csv"
    if not interim.exists():
        oc = z_in + z_occ
        write_table(interim, ncs, onet, short, np.argsort(-oc, axis=1, kind="stable"),
                    {"OC_점수": oc, "IN_점수": base, "직업_점수": occ})
        say(f"중간 연계표 저장: {interim.relative_to(ROOT)}")

    # 4. 재순위
    task_text = build_texts("onet_B", ncs, onet)
    starts = [s for s in range(0, n, RERANK_CHUNK) if not (PARTS / f"part_{s:06d}.npy").exists()]
    if starts:
        from sentence_transformers import CrossEncoder
        from e5_embedder import pick_device
        cross = CrossEncoder(RERANKER, device=pick_device(), max_length=256)
        for s in starts:
            started = time.time()
            rows = range(s, min(s + RERANK_CHUNK, n))
            prob = np.asarray(cross.predict([(texts["B"][i], task_text[j]) for i in rows for j in short[i]],
                                            batch_size=16), np.float32).reshape(len(rows), SHORT)
            np.save(PARTS / f"part_{s:06d}.npy", prob)
            say(f"재순위 {rows[-1] + 1:,} / {n:,} 수행준거 ({time.time() - started:.0f}초)")
        del cross

    # 5. 최종 연계표
    prob = np.concatenate([np.load(PARTS / f"part_{s:06d}.npy") for s in range(0, n, RERANK_CHUNK)]).astype(np.float64)
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    logit = np.log(clipped / (1 - clipped))
    fn = z_in + z_occ + zrow(logit)
    np.savez(WORK / "scores.npz", short=short, IN=base, 직업=occ.astype(np.float32), RB=logit.astype(np.float32),
             FN=fn.astype(np.float32))
    final = CROSS / "ncs_onet_task_FN.csv"
    write_table(final, ncs, onet, short, np.argsort(-fn, axis=1, kind="stable"),
                {"FN_점수": fn, "IN_점수": base, "직업_점수": occ, "재순위_점수": logit})
    say(f"최종 연계표 저장: {final.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
