# VEWD-2026 O*NET–NCS Crosswalk

* O\*NET과 NCS를 의미 유사도로 연결하는 작업입니다.

* DISCLAIMER: THIS IS NOT AUTHORIZED BY US Employment and Training Administration, National Center for O\*NET Development nor ROK Mnistry of Employment of Labor, Human Resources Development Service of Korea


NCS의 "수행 준거"와 O*NET의 "Task Statements"를 BERT 알고리즘 계열 기반 모델을 활용해 연결합니다.
기본적으로 crosswalk between ESCO and O\*NET이 사용한 방식을 차용했으며, 자세한 사항은 아래 링크를 확인해 주세요.
([esco.ec.europa.eu/en/use-esco/other-crosswalks](https://esco.ec.europa.eu/en/use-esco/other-crosswalks))
![1794728660359](image/README/1794728660359.png)
(Figure 출처: [esco.ec.europa.eu/system/files/2022-12/ONET%20ESCO%20Technical%20Report.pdf](<https://esco.ec.europa.eu/system/files/2022-12/ONET%20ESCO%20Technical%20Report.pdf>))

본 작업의 임베딩 모델은[`multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)를 사용합니다. 교차 언어 검색·유사도에서 안정적인 성능을 보여주는 BERT 계열 모델로 평가받고 있습니다.

multilingual-e5-large는 언어가 달라도 뜻이 비슷한 문장을 가까운 벡터(1,024차원)로 바꿉니다. 따라서 번역 없이 다음 세 단계로 연결할 수 있습니다.

1. 수행준거와 Task를 각각 벡터로 변환(Transform).
2. 수행준거마다 가장 가까운 Task 상위 k개를 찾는다.
3. 순위와 점수로 후보를 걸러 내고, 표본을 사람이 검증한다.
4. 3.의 과정을 통해 만든 표본을 바탕으로 모델을 학습시킨다.
5. 모델의 성능을 확인해 가장 fit한 모델을 이후 분석에 사용한다.

## 1. 임베딩 모델 설치 (macOS, Apple Silicon 기준)

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

(추천하지 않음) 메모리가 부족하면 반정밀 모드를 씁니다.

```bash
.venv/bin/python check_model.py --fp16
```

## VS Code에서 작업하기

1. VS Code에서 이 폴더를 엽니다 (`파일 > 폴더 열기`).
2. 오른쪽 아래에 뜨는 권장 확장(Python, Jupyter) 설치 알림을 수락합니다.
3. 인터프리터는 `.venv`가 자동으로 선택됩니다. 선택되지 않으면 `Cmd+Shift+P` →
   `Python: Select Interpreter` → `.venv`를 고릅니다.
4. `.py` 파일은 오른쪽 위 실행 버튼으로, 노트북(`.ipynb`)은 커널을 `.venv`로 골라 실행합니다.

## 코드에서 사용하기(예시)

```python
from e5_embedder import load_model, embed

model = load_model()                      # 메모리가 부족하면 load_model(fp16=True)
onet = embed(model, ["Registered Nurses: Assess patient health problems..."])
ncs = embed(model, ["간호: 환자의 건강 문제를 사정하고...", "용접: 금속 재료를 접합..."])

print(onet @ ncs.T)                       # 코사인 유사도 (행: O*NET, 열: NCS)
```

## 알아 둘 점

- **접두어**: e5는 입력 앞에 `query: `가 필요하며 `embed()`를 자동으로 붙입니다.
- **점수 범위**: e5의 유사도는 대부분 0.7~0.9 사이에 몰립니다. 절대값이 아니라 순위로 판단해야 합니다.
- **짧은 명칭의 한계**: 직무 설명문을 함께 넣어야 구분이 가능합니다. 직업명으로만 비교할 경우 점수 차이가 작습니다.
- **입력 길이**: 512 토큰을 넘는 부분은 잘립니다.
- **메모리**: batch_size=""의 ""에 컴퓨팅 환경의 메모리 용량을 입력합니다. 8GB 메모리를 사용한다면`batch_size=8`으로 작성하고, 다른 앱을 닫고 실행하는 것이 좋습니다.
