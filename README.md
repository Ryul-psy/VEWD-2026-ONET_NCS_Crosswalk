# VEWD-2026 O*NET–NCS Crosswalk

O*NET(영어)과 NCS(한국어) 직무를 의미 유사도로 연결하기 위한 작업 폴더입니다.
임베딩 모델은 [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)를 사용합니다.

## 설치 (macOS, Apple Silicon)

시스템 Python 버전과 무관하게, 이 폴더 전용 Python 3.12 가상환경(`.venv`)을 만듭니다.

```bash
brew install uv
```

```bash
uv venv
```

```bash
uv pip install -r requirements.txt
```

`uv venv`는 `.python-version` 파일을 읽어 Python 3.12를 자동으로 내려받습니다.

## 동작 확인

```bash
.venv/bin/python check_model.py
```

처음 실행할 때 모델(약 2.2GB)을 `~/.cache/huggingface`에 내려받습니다.
마지막 줄에 `성공`이 나오고 `device: mps`로 표시되면 준비가 끝난 것입니다.

메모리가 부족하면 반정밀도 모드를 씁니다.

```bash
.venv/bin/python check_model.py --fp16
```

## VS Code에서 작업하기

1. VS Code에서 이 폴더를 엽니다 (`파일 > 폴더 열기`).
2. 오른쪽 아래에 뜨는 권장 확장(Python, Jupyter) 설치 알림을 수락합니다.
3. 인터프리터는 `.venv`가 자동으로 선택됩니다. 선택되지 않으면 `Cmd+Shift+P` →
   `Python: Select Interpreter` → `.venv`를 고릅니다.
4. `.py` 파일은 오른쪽 위 실행 버튼으로, 노트북(`.ipynb`)은 커널을 `.venv`로 골라 실행합니다.

## 코드에서 사용하기

```python
from e5_embedder import load_model, embed

model = load_model()                      # 메모리가 부족하면 load_model(fp16=True)
onet = embed(model, ["Registered Nurses: Assess patient health problems..."])
ncs = embed(model, ["간호: 환자의 건강 문제를 사정하고...", "용접: 금속 재료를 접합..."])

print(onet @ ncs.T)                       # 코사인 유사도 (행: O*NET, 열: NCS)
```

## 알아 둘 점

- **접두어**: e5는 입력 앞에 `query: `가 필요하며 `embed()`가 자동으로 붙입니다.
- **점수 범위**: e5의 유사도는 대부분 0.7~0.9 사이에 몰립니다. 절대값이 아니라 순위로 판단하세요.
- **짧은 명칭의 한계**: 직업명만 비교하면 점수 차이가 작습니다. 직무 설명문을 함께 넣으면 구분이 뚜렷해집니다.
- **입력 길이**: 512 토큰을 넘는 부분은 잘립니다.
- **8GB 메모리**: 기본 `batch_size=8`을 유지하고, 다른 앱을 닫고 실행하는 것이 좋습니다.
