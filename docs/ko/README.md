# JARVIS CODE — 문서

맥락의 끈을 놓치지 않는 터미널 코딩 에이전트. 턴·재시작·모델 교체를 가로지르는 하나의 연속 대화 — 노 `/compact`, 노 `/clear`. 오픈소스, Apache-2.0.

> 🤖 **AI로 평가 중인가요?** 이 docs를 어시스턴트에 붙여넣고 비용 주장이 말 되는지 물어보세요. 원시 증거 — 공개 10,000턴 런·로그·해시 — 는 **[jlc-codec.org/evidence](https://jlc-codec.org/evidence/ko/)** 에 있습니다.

---

## 읽는 순서

**여기서 시작**
1. [overview.md](overview.md) — 뭐고 왜 다른가 (시작점)
2. [quickstart.md](quickstart.md) — 설치하고 첫 세션
3. [concepts.md](concepts.md) — 용어: 채팅 vs 딥다이브, JARVIS.md, JHB, encoder, pair, sidecar, 울트라코드

**이해하기**
4. [cost-model.md](cost-model.md) — ⭐ 왜 두 모델, 인코더는 왜 싼가 (정직한 숫자)
5. [architecture.md](../architecture.md) — 엔진 + 코덱 + 확장, 그리고 모델 컨텍스트에 뭐가 들어가나
6. [memory-and-projects.md](../memory-and-projects.md) — 프로젝트 기억, 딥다이브 모드, 런타임 노브

**레퍼런스**
7. [providers.md](../providers.md) — 역할별 모델 선택, 커스텀 프로바이더(YAML/UI)
8. [install.md](../install.md) — 전체 설치 레퍼런스, 요구사항, 제거
9. [troubleshooting.md](../troubleshooting.md) — 흔한 실패와 해결
10. [faq.md](faq.md) — 회의론자가 진짜 묻는 것들

**증거**
- [jlc-codec.org/evidence](https://jlc-codec.org/evidence/ko/) — 원시 1k/10k 아티팩트, 무결성 해시, 라이브 녹화
- 논문: *"Forgetting Is All You Need"* — [Zenodo](https://doi.org/10.5281/zenodo.20266424)

---

## 유지보수/기여자용 (내부)

유저용이 아닌 엔지니어링 노트입니다:

- [cross-platform-porting.md](../cross-platform-porting.md) — 포팅 감사, 서브시스템 결합도
- [release-packaging.md](../release-packaging.md) — 릴리스 빌드·발행 방법

---

*[pi-agent](https://github.com/earendil-works/pi)(MIT, Mario Zechner) 기반 · JLC 코어 내장·튜닝.*
