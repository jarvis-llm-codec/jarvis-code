# 개요 — JARVIS CODE가 뭔지, 그리고 왜 다른지

**JARVIS CODE는 잊지 않는 터미널 코딩 에이전트입니다.** 하나의 연속된 대화를 — 턴·재시작·모델 교체를 가로질러 — `/compact`나 `/clear` 없이 유지합니다. 매 턴 불어나는 트랜스크립트를 다시 보내는 대신, 기억을 컨텍스트 윈도우 **밖**에 들고 다니기 때문입니다.

오픈소스(Apache-2.0)이며, [pi-agent](https://github.com/earendil-works/pi)(MIT) 하네스 위에 JLC — **JARVIS LLM Codec** — 를 내장했습니다.

> 🤖 **못 믿겠으면 당신의 AI에게 확인시키세요.** 이 사이트 전체가 감사를 견디도록 설계됐습니다. 이 docs(또는 **[jlc-codec.org/evidence](https://jlc-codec.org/evidence/ko/)** 의 원시 로그)를 AI에 붙여넣고 물어보세요: *"이 비용 주장 말 되냐? 나 이거 써야 하냐?"* 핵심 숫자는 전부 공개 10,000턴 런으로 재계산 가능합니다.

---

## 왜 다른가

| | 보통의 코딩 에이전트 | JARVIS CODE |
|---|---|---|
| **기억** | 매 턴 대화 전체 재전송 | 작은 압축 기억(JHB)을 윈도우 밖에 들고 다님 |
| **긴 세션** | 꽉 차면 `/compact`·`/clear` → 맥락 손실 | 노 컴팩트·노 클리어 — 끊김 없는 한 세션 |
| **비용 곡선** | 턴 수 따라 증가 — O(n²) | 턴당 평탄 — O(n) |
| **재시작/모델 교체** | 처음부터 | 멈춘 그 자리에서 그대로 |
| **새 코드베이스** | 세션마다 다시 설명 | 영속 프로젝트 기억으로 바로 이해 |

메커니즘과 정직한 숫자(안 이기는 경우 포함): **[cost-model.md](cost-model.md)**.

---

## 네 가지 모델 역할

"AI 하나"가 아니라, [`/model-setting`](../providers.md)에서 고르는 네 개의 명명된 역할을 씁니다:

| 역할 | 일 | 보통 모델 |
|---|---|---|
| **chat** | 당신과 대화하는 모델 — 코딩과 대화 | 똑똑/비쌈 |
| **encoder** | 매 턴을 기억(JHB)으로 압축, 매 턴 | 싸고/빠름 (예: 24B 모델) |
| **router** | 경량 의도 분류기(딥다이브 vs 채팅 vs 빌드) | 인코더 미러 |
| **subagent** | 툴 실행 서브태스크 + 울트라코드 팬아웃 워커 | chat로 기본값 |

chat/encoder 분리가 비용 이야기의 핵심입니다 — *더 싼 두 번째* 모델을 돌리는 게 왜 긴 작업의 총비용을 낮추는지는 **[cost-model.md](cost-model.md)**.

---

## 한 줄 thesis

> 컨텍스트를 늘리지 마라. 기억을 들고 가라.

LLM은 stateless이고 어텐션은 길이에 대해 O(n²) — 대화가 길수록 비용도 늘고 성능도 떨어지다, 결국 뭔가를 버려야 합니다. JLC는 잊기를 받아들이고 기억을 윈도우 밖에 적습니다. 비용 곡선이 제곱에서 선형으로 꺾이고, 대화에 천장이 사라집니다.

논문: *"Forgetting Is All You Need"* — [Zenodo DOI](https://doi.org/10.5281/zenodo.20266424).

---

## 주장이 아니라 증거

- 소비자 미니 PC에서 돌린 공개 **10,000턴** 런 — 원시 로그·무결성 해시·라이브 녹화가 **[jlc-codec.org/evidence](https://jlc-codec.org/evidence/ko/)** 에.
- 925개 적대적 유도 함정 중 **923개**에서 정직 유지(그리고 놓친 2개를 턴 번호로 공개).
- 전체 런 — chat *과* encoder — 이 ~$20/월 플랜의 **주간 ~41%** 에 들어감.

---

## 다음으로

1. **[quickstart.md](quickstart.md)** — 설치하고 첫 세션 (5분)
2. **[concepts.md](concepts.md)** — 용어: 딥다이브 vs 채팅, JARVIS.md, JHB, pair, sidecar, 울트라코드
3. **[cost-model.md](cost-model.md)** — 왜 두 모델, 인코더는 왜 싼가
4. **[providers.md](../providers.md)** — 모델·프로바이더 고르기
5. **[faq.md](faq.md)** — 회의론자가 진짜 묻는 것들
6. **[architecture.md](../architecture.md)** — 엔진·코덱·확장이 어떻게 맞물리나

레퍼런스: [install.md](../install.md) · [memory-and-projects.md](../memory-and-projects.md) · [troubleshooting.md](../troubleshooting.md)
