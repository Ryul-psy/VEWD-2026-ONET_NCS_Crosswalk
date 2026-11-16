"""수행준거와 Task를 A·B 방식으로 임베딩해 저장한다.

    python embed_texts.py

결과: data-ONET-NCS/batch1/embeddings/{onet_A,onet_B,ncs_A,ncs_B}.npy
      (원자료와 같은 행 순서, 길이 1로 정규화, float16)

M2 맥북 에어 기준 2시간쯤 걸림. 중간에 멈춰도 다시 실행하면 끝난 부분부터 이어서 한다.
진행 상황은 data-ONET-NCS/batch1/embed_progress.txt에 부분(4,000문장)마다 갱신된다.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from e5_embedder import embed, load_model
from linkdata import EMB, KINDS, OUT, build_texts, read_ncs, read_onet

PART_SIZE = 4000   # 이 문장 수마다 중간 저장
BATCH_SIZE = 32
PROGRESS = OUT / "embed_progress.txt"


def write_progress(plan: dict, done: dict, worked: int, started: float, state: str) -> None:
    """진행 상황을 텍스트 파일로 남긴다. 남은 시간은 글자 수 기준으로 추정한다."""
    total = sum(chars for _, chars in plan.values())
    finished = sum(chars for _, chars in done.values())
    elapsed = time.time() - started
    lines = [
        "임베딩 진행 상황",
        f"갱신: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"상태: {state}",
        f"전체: {finished / total:.1%}",
    ]
    if state == "진행 중" and worked:
        left = elapsed * (total - finished) / worked
        lines.append(f"남은 시간: 약 {left / 60:.0f}분 (예상 종료 {time.strftime('%H:%M', time.localtime(time.time() + left))})")
    lines += ["", f"{'대상':<8}{'끝난 문장':>12}{'전체 문장':>12}{'진행률':>8}"]
    for kind, (count, chars) in plan.items():
        n, c = done[kind]
        lines.append(f"{kind:<8}{n:>14,}{count:>14,}{c / chars:>10.0%}")
    lines += ["", "onet = O*NET Task, ncs = NCS 수행준거, A = 문장만, B = 상위 분류 이름 포함",
              "문장 수는 서로 다른 문장 기준이며, 짧은 문장부터 계산하므로 뒤로 갈수록 느려집니다."]
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parts_dir = EMB / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ncs, onet = read_ncs(), read_onet()
    model = None

    # 진행 상황 계산용: 대상별 서로 다른 문장 수와 글자 수
    # 길이가 같은 문장의 순서도 실행마다 같아야 한다 (다르면 이어서 실행할 때 벡터가 뒤바뀐다)
    uniques = {kind: sorted(set(build_texts(kind, ncs, onet)), key=lambda t: (len(t), t)) for kind in KINDS}
    plan = {kind: (len(u), sum(map(len, u))) for kind, u in uniques.items()}
    progress = {kind: (0, 0) for kind in KINDS}
    began, worked = time.time(), 0

    for kind in KINDS:
        final = EMB / f"{kind}.npy"
        texts = build_texts(kind, ncs, onet)
        if final.exists() and len(np.load(final, mmap_mode="r")) == len(texts):
            print(f"{kind}: 이미 있음", flush=True)
            progress[kind] = plan[kind]
            write_progress(plan, progress, worked, began, "진행 중")
            continue

        # 서로 다른 문장만, 길이가 비슷한 것끼리 묶어 계산한다
        unique = uniques[kind]
        n_parts = -(-len(unique) // PART_SIZE)
        start_time, done = time.time(), 0
        for p in range(n_parts):
            part_file = parts_dir / f"{kind}_{p:04d}.npy"
            chunk = unique[p * PART_SIZE:(p + 1) * PART_SIZE]
            n_done, c_done = progress[kind]
            progress[kind] = (n_done + len(chunk), c_done + sum(map(len, chunk)))
            if part_file.exists() and len(np.load(part_file, mmap_mode="r")) == len(chunk):
                continue
            if model is None:
                model = load_model()
            vecs = embed(model, chunk, batch_size=BATCH_SIZE)
            if not np.isfinite(vecs).all():
                print(f"{kind} {p}: 벡터에 NaN/inf가 있습니다.")
                return 1
            np.save(part_file, vecs.astype(np.float16))
            done += len(chunk)
            rate = done / (time.time() - start_time)
            left = (len(unique) - (p + 1) * PART_SIZE) / rate / 60
            print(f"{kind}: {p + 1}/{n_parts} 부분, 초당 {rate:.0f}문장, 남은 시간 약 {max(left, 0):.0f}분",
                  flush=True)
            worked += sum(map(len, chunk))
            write_progress(plan, progress, worked, began, "진행 중")

        # 원자료 행 순서로 펼쳐 저장한다
        position = {text: i for i, text in enumerate(unique)}
        vectors = np.concatenate([np.load(parts_dir / f"{kind}_{p:04d}.npy") for p in range(n_parts)])
        np.save(final, vectors[[position[t] for t in texts]])
        for p in range(n_parts):
            (parts_dir / f"{kind}_{p:04d}.npy").unlink()
        print(f"{kind}: 저장 {final.name} {len(texts):,}행 (서로 다른 문장 {len(unique):,}개)", flush=True)

    write_progress(plan, progress, worked, began, "완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())