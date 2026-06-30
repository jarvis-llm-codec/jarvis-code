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
| `bge-m3` 임베딩 모델 — *선택* | ~2.3 GB 다운로드 · 디스크 ~4.3 GB |
| 전제조건 (Node 20+, Python 3.10+, Git, MSVC) — *없을 때만* | ~0.5 GB |

**모델 없이 ≈ 2 GB, 모델 포함 ≈ 6 GB** (전제조건은 없을 때만 추가). 모델은 로컬 회상용이고 선택 — `JARVIS_CODE_NO_MODEL_PRELOAD=1`로 생략. 메모리 데이터(`~/.jarvis-code`)는 작게 시작해 천천히 늘어남.

---

## 2. 실행

```
jarvis
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
