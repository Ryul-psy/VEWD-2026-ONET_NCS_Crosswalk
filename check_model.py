"""설치와 모델 동작을 확인하는 스크립트.

    python check_model.py          # 기본(fp32)
    python check_model.py --fp16   # 메모리 절약 모드
"""

import argparse
import sys
import time

import numpy as np
import sentence_transformers
import torch

from e5_embedder import MODEL_NAME, embed, load_model, pick_device

ENGLISH = ["Registered Nurses", "Welders, Cutters, Solderers, and Brazers", "Accountants and Auditors"]
KOREAN = ["간호사", "용접공", "회계사"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="반정밀도로 불러와 메모리를 절약")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"sentence-transformers {sentence_transformers.__version__}")
    print(f"device: {pick_device()} | model: {MODEL_NAME} | fp16: {args.fp16}")

    start = time.time()
    model = load_model(fp16=args.fp16)
    print(f"모델 로드: {time.time() - start:.1f}초")

    start = time.time()
    en = embed(model, ENGLISH)
    ko = embed(model, KOREAN)
    print(f"임베딩: {time.time() - start:.1f}초 | 벡터 크기: {en.shape[1]}")

    sim = en @ ko.T
    if not np.isfinite(sim).all():
        print("실패: 유사도에 NaN/inf가 있습니다. --fp16 없이 다시 실행해 보세요.")
        return 1

    print("\n코사인 유사도 (행: 영어, 열: 한국어)")
    print(" " * 42 + "".join(f"{k:>8}" for k in KOREAN))
    for name, row in zip(ENGLISH, sim):
        print(f"{name:<42}" + "".join(f"{v:>10.3f}" for v in row))

    if (sim.argmax(axis=1) == np.arange(len(ENGLISH))).all():
        print("\n성공: 각 영어 직업명이 같은 뜻의 한국어 직업명과 가장 가깝습니다.")
        return 0
    print("\n실패: 짝이 맞지 않는 항목이 있습니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
