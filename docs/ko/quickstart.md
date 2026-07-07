# 빠른 시작 — 5분 안에 첫 세션

제로에서 작동하는 코딩 세션까지. 전체 설치 레퍼런스(요구사항·제거·폴더 위치)는 [install.md](../install.md), 보고 있는 게 뭔지는 [concepts.md](concepts.md).

---

## 1. 설치

**Windows** (현재 지원) — PowerShell에서:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

**macOS · Linux** — 인스톨러 마무리 중 (**곧 지원 예정**). 그 전까지는 소스에서 설치 — [install.md](../install.md) 참고.

### 뭐가 설치되고 얼마나 큰가

전부 `%LOCALAPPDATA%\JARVIS-Code` 아래 설치. 새 머신 기준 대략 용량:

| 구성 | 크기 |
|---|---|
| 앱 + Node 의존성 | ~580 MB |
| Python 사이드카 (PyTorch CPU 포함) | ~1.3 GB |
| NVIDIA CUDA PyTorch — `nvidia-smi`가 감지될 때만 | ~2.7 GB 다운로드 |
| `bge-m3` 임베딩 모델 — **회상 담당 (필수)** | ~2.3 GB 다운로드 · 디스크 ~4.3 GB |
| 전제조건 (Node 20+, Python 3.10+, Git, MSVC) — *없을 때만* | ~0.5 GB |

**CPU 설치는 총 ≈ 6 GB, NVIDIA 설치는 여기에 ~2.7 GB가 더해질 수 있음** (전제조건은 없을 때만 추가). 회상은 **BM25 + bge-m3** 하이브리드(키워드 + 시맨틱)라 모델은 선택이 아니라 **핵심** — `JARVIS_CODE_NO_MODEL_PRELOAD=1`은 다운로드를 첫 사용으로 **미룰 뿐** 의존성을 없애진 않음. 메모리 데이터(`~/.jarvis-code`)는 작게 시작해 천천히 늘어남.

`nvidia-smi`가 있고 정상 실행되면 Windows 인스톨러가 PyTorch의 `cu126` 휠 인덱스에서 CUDA PyTorch를 먼저 설치한 뒤 사이드카 의존성을 설치합니다. CPU PyTorch를 강제하려면 설치 전에 `JARVIS_CODE_CPU_ONLY=1`을 설정하세요.

---

## 2. 실행

```
jarvis
```

수동/소스 설치처럼 사이드카 의존성이 미리 설치되지 않은 경우 첫 실행이 몇 분 걸릴 수 있습니다. JARVIS는 처음 세 번 실행 동안 사이드카 창을 자동으로 보여줍니다. 문제 확인이나 재설치 검증 때는 이렇게 시작하세요:

```
jarvis --sidecar-window
```

첫 실행 시 안내해줍니다:

1. **프로바이더 로그인** — 모델 프로바이더 연결(예: Claude 구독, 또는 Ollama Cloud 키). 손으로 편집할 설정 파일 없음.
2. **모델 역할** — 역할별 모델 선택. 중요한 둘:
   - **chat** — 코딩하는 똑똑하고 비싼 모델
   - **encoder** — 기억을 담당하는 싸고 빠른 모델 (매 턴 동작)

   좋은 시작 조합: **chat**에 프론티어 모델, **encoder**에 ~24B 모델(예: `ollama-cloud/devstral-small-2:24b`). 이 분리가 왜 돈을 아끼는지: [cost-model.md](cost-model.md). 나중에 바꾸려면: `/model-setting`.

이후 실행은 전부 건너뜁니다 — 기억하니까요.

---

## 3. 프로젝트 열기

코드 저장소 *안에서* `jarvis`를 실행하세요. 에이전트는 프로젝트별 기억 파일(**JARVIS.md**)을 유지해, 작업하며 당신의 코드베이스를 배우고 — 다음 세션에도 기억합니다. 프로젝트 전환에 재설정·재설명이 필요 없습니다([concepts.md](concepts.md)).

---

## 4. 작업 — 그리고 계속

그냥 대화하세요: 묻고, 변경을 설명하고, 빌드시키세요. 다른 에이전트가 안 하는 두 가지를 보게 됩니다:

- **`/compact`나 `/clear`로 멈추지 않습니다.** 대화가 계속됩니다 — 10,000턴으로 입증([증거](https://jlc-codec.org/evidence/ko/)).
- **닫고 내일 다시 와도** — 멈춘 그 자리에서, *왜* approach X를 버렸는지까지 기억한 채 이어갑니다.

일상 대화엔 가벼운 추론을, 코딩엔 추론을 끝까지 — 컨텍스트 태울 걱정 없이(기억이 윈도우 밖에 사니까).

---

## 다음 단계

- [concepts.md](concepts.md) — JARVIS.md·JHB·"pair"·sidecar가 실제로 뭔지
- [cost-model.md](cost-model.md) — 두 모델 구성이 긴 작업에서 왜 더 싼지
- [providers.md](../providers.md) — 다른 프로바이더/로컬 모델 연결
- [troubleshooting.md](../troubleshooting.md) — 시작이 안 될 때
