"""multilingual-e5-large 임베딩 도우미.

사용 예:
    from e5_embedder import load_model, embed

    model = load_model()
    vecs = embed(model, ["Registered Nurses", "간호사"])
    print(vecs @ vecs.T)  # 코사인 유사도 활용 유사도 측정
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"


def pick_device() -> str:
    """Apple Silicon이면 mps, NVIDIA GPU가 있으면 cuda, 아니면 cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(
    device: str | None = None,
    fp16: bool = False,
    max_seq_length: int = 512,
) -> SentenceTransformer:
    """모델을 불러온다. 처음 실행할 때 약 2.2GB를 내려받는다.
    fp16=True는 메모리를 절반으로 줄이지만 cpu에서는 쓰지 않는다.
    """
    device = device or pick_device()
    model = SentenceTransformer(MODEL_NAME, device=device)
    if fp16 and device != "cpu":
        model.half()
    model.max_seq_length = max_seq_length  # e5의 최대 입력 길이는 512 토큰
    return model


def embed(
    model: SentenceTransformer,
    texts: list[str],
    prefix: str = "query: ",
    batch_size: int = 8,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """문장들을 정규화된 벡터(행 단위)로 바꾼다.

    e5는 입력 앞에 접두어가 필요하다. 직무 대 직무처럼 대칭적인 유사도 비교는
    양쪽 모두 "query: "를 쓰고, 질문으로 문서를 찾는 검색에서는 문서 쪽에
    prefix="passage: "를 준다.
    """
    vecs = model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return vecs.astype(np.float32)
