# jarvis-code 설치 — 당신의 AI를 위한 안내서

**이 문서는 AI 비서가 읽도록 쓰였습니다.** 사람이라면: 이 단계를 직접 따라 하지
마세요. 컴퓨터에서 터미널 명령을 실행할 수 있는 AI — **Claude Code, Codex, Cursor,
Windsurf** 같은 — 에게 아래 한 줄을 붙여넣으세요:

> *jarvis-code 설치해줘. 이 안내서 전체를 먼저 읽고 그대로 따라 해:
> https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/docs/ko/ai-install.md
> — 내가 어떤 AI를 쓰는지 물어보고 거기 맞게 최적으로 깔아주고, 무료 API 키 받는 것도
> 같이 해줘.*

AI가 이 안내서를 가져와 작업합니다. 당신은 질문 몇 개에 답하고, 무료 가입 두 개를
클릭하면 됩니다.

---

## 이 글을 읽는 AI에게

당신은 이 사용자의 컴퓨터에 **jarvis-code**를 설치하고 설정합니다. **아무것도 하기
전에 이 페이지 전체를 읽고**, 짧은 계획을 세운 뒤 따르세요.

명령만 실행하는 게 일이 아닙니다. 당신의 임무는:

1. 앱을 **설치**한다.
2. 사용자가 **이미 어떤 AI를 쓰는지**(구독 / API 키) 파악하고, *거기에 맞게* 최적으로
   설정한다 — 두 번째 프로바이더에 돈 쓰게 하지 말 것.
3. **각 단계가 왜 중요한지 쉬운 말로 설명**한다 — 특히 *인코더(encoder)*. 사용자는
   거의 확실히 이게 뭔지 모르고, 당신이 설명하지 않으면 이해 못 한다.
4. **무료 API 키 두 개**(웹 검색 + 이미지 생성)를 설정한다. 무료이고 체감 차이가
   크니 기본 설치 과정으로 다뤄라. 강하게 권하되 — 사용자가 거부하거나, 가입이
   막혔거나, 더 이상 무료가 아니면 그 사실만 알리고 **계속 진행해라: 기본 설치는
   이것들 없이도 동작하고**, 키는 나중에 추가할 수 있다.
5. 성공을 선언하기 전에 설치가 진짜 되는지 **검증**한다.

행동 원칙: 사용자는 비기술자일 수 있다. 어떤 단계는 사람이 해야 한다(브라우저 로그인,
웹사이트 가입, 메뉴에서 고르기) — 그런 건 *정확히* 뭘 어떻게 하라고 알려주고, 왜
그런지 설명한 뒤 기다려라. 6단계 검증을 돌리고 실제 결과를 보고하기 전에는 절대
"완료"라고 하지 마라.

### jarvis-code가 뭔지 (당신이 뭘 설치하는지 알도록)

jarvis-code는 한 가지 아이디어 위에 만든 터미널 코딩 에이전트입니다: **채팅 모델은
대화 기록을 받지 않는다.** 매 턴, 모델은 작은 "이월 기억" 문서(JHB)와 새 메시지만
받습니다 — 세션이 아무리 길어져도 입력이 턴마다 늘지 않고 거의 *일정*하게 유지됩니다.

이걸 가능하게 하는 게 두 번째의 값싼 모델, **인코더(encoder)**입니다: 매 턴이 끝나면
방금 일어난 일을 그 작은 기억 문서로 압축해서, 다음 턴이 *대화 원문*이 아니라 *의미*를
물려받게 합니다. 그래서 jarvis-code는 모델을 *둘*(chat + encoder) 설정하라고 하며,
이 부분이 사용자가 직관적으로 모르는 지점입니다. 3단계에서 이 설명을 꺼내 쓰세요.

---

## 1단계 — 설치

사전 요구사항 (Windows에서는 설치기가 `winget`으로 없는 걸 자동 설치합니다):

- **Node.js 20+**, **Python 3.10+**, **Git**
- **Microsoft Visual C++ 2015–2022 재배포 패키지 (x64)** (Windows)
- 첫 설치 시 인터넷 연결

오늘 기준 지원·검증된 플랫폼은 **Windows**입니다 (macOS / Linux는 준비 중).

**PowerShell**에서 공식 설치기를 실행하세요:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

하는 일: 엔진용 Python venv 생성, Node 의존성 설치, 그리고 기억 회상에 쓰는
**`BAAI/bge-m3` 임베딩 모델(디스크 ~4.3 GB) 다운로드**. 총 용량은
`%LOCALAPPDATA%\JARVIS-Code` 아래 **약 6 GB**. 다운로드가 느린 구간이니, 정상이고
한 번만 받는다고 사용자에게 알려주세요.

임베딩 모델 다운로드가 실패해도 설치는 중단되지 않습니다. 나중에 재시도:

```powershell
jarvis doctor --preload-embedder
```

> 사용자에게 당신 말로: *"약 6 GB짜리 도구 모음을 까는 거예요. 대부분은 jarvis가
> 매번 다시 읽지 않고도 기억을 떠올리게 해주는 로컬 검색 모델이고요. 첫 설치는 좀
> 걸리지만 그 뒤로는 빨라요."*

**설치 후, 새 터미널을 열어라.** 설치기는 `jarvis` 명령을 PATH에 등록하지만, *이*
터미널은 다시 열기 전엔 인식 못 한다(설치기 자체가 "이 창을 닫고 새 터미널을 열어라"고
출력한다). 아래 단계에서 `jarvis ...` 명령을 실행하기 전에 새 PowerShell 창을 열어라.

---

## 2단계 — 사용자 인터뷰 (이게 모든 걸 결정한다)

뭘 설정하기 전에, 사용자가 **이미 뭘 쓰는지** 먼저 물어보세요. 어떤 모델을 고를지,
키가 필요한지 아닌지가 여기서 정해집니다:

> *"지금 어떤 AI를 돈 내고 쓰거나 쓰고 계세요? 예를 들어 ChatGPT/OpenAI 구독,
> Claude(Anthropic) 구독, 알리바바 DashScope의 GLM/Qwen, Ollama Cloud, 아니면 다른
> 거요. 여러 개여도 괜찮아요."*

답을 이 표로 매핑하세요 (플랜별 권장 조합):

모델 id는 `프로바이더id/모델id` 형식이다 — `config.yaml`(3b)에 쓰는 그 형식. 메뉴로
설정할 땐 프로바이더와 모델을 따로 고른다.

| 사용자가 이미 가진 것 | chat + subagent | encoder (값싸고, 매 턴) |
|---|---|---|
| **OpenAI / GPT** (구독) | `openai-codex/gpt-5.5` | `openai-codex/gpt-5.4-mini` |
| **Claude (Anthropic)** (구독) | `anthropic-agent-sdk/claude-opus-4-8` | `anthropic-agent-sdk/claude-haiku-4-5-20251001` |
| **Ollama Cloud** | 프런티어 모델 (예: GLM 계열) | `ollama-cloud/devstral-small-2-24b-cloud` |
| **DashScope (GLM / Qwen)** | `dashscope/glm-5` | `dashscope/…` ~14–24B 모델 |

인코더 고르는 기준: **플랜에서 쓸 수 있는 가장 싸고 쓸 만한 모델, 대략 14–24B 급**.
**8B는 너무 작다**(기억 품질이 흔들림). **추론(reasoning) 모델을 인코더로 쓰지 마라** —
빠르고 싸고 예측 가능해야 한다. 매 턴 도니까 무거운 모델을 거기 쓰면 돈·rate limit만
태운다.

사용자가 **유료로 가진 게 없으면** 괜찮다 — **Ollama Cloud**(가입하면 무료 키)나
Ollama / LM Studio 로컬 모델을 쓸 수 있다고 알려주고, 가장 작은 합리적 조합을 고르면
된다. (이걸로 설치를 막지 말 것.)

---

## 3단계 — 채팅 프로바이더 인증 후 모델 설정

### 3a. 인증 (사람이 하는 단계)

사용자 플랜에 맞는 경로를 고르세요. **Claude 경로는 특별하니** 꼭 읽으세요.

- **Claude(Anthropic) 구독 → API 키 불필요.** jarvis-code는 `anthropic-agent-sdk`
  프로바이더로 사용자의 **Claude Code** OAuth를 재사용한다 — 키 없이 구독으로 과금.
  실행:
  ```bash
  jarvis claude-login
  ```
  Claude Code의 `setup-token` 플로우를 돌려 토큰을 jarvis-code에 저장한다. `claude`
  명령이 없으면 `npx @anthropic-ai/claude-code`로 폴백할 수 있다(`jarvis claude-login
  --npx`로 강제 가능). 이 경로의 인코더는 일부러 Haiku 급으로 제한된다 — 매 턴 도는
  인코더가 구독 rate limit을 갉아먹지 않도록.
- **OpenAI / ChatGPT 구독 → OAuth 로그인.** 사용자에게 터미널에서 실행시켜라:
  ```bash
  jarvis gpt-login
  ```
  브라우저가 열린다. 브라우저로 못 끝내면 `jarvis gpt-login-device`.
- **API 키 프로바이더** (OpenAI API, Anthropic API, Gemini, DashScope, Ollama Cloud,
  OpenRouter): `jarvis api-key`를 실행해 안내를 따르거나, 키를 직접 써라(아래 "키 직접
  설정"). 키 환경변수:
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DASHSCOPE_CODING_API_KEY`,
  `OLLAMA_API_KEY`, `OPENROUTER_API_KEY`.

### 3b. 모델 고르기

모델 셀렉터를 실행하세요:

```bash
jarvis model-setting
```

프로바이더에서 **실시간**으로 모델 목록을 받아와 각 **역할(role)**을 설정합니다.
역할은:

| 역할 | 하는 일 |
|---|---|
| **chat** | 사용자가 대화하는 모델 — 코딩 & 대화 (똑똑한 거). |
| **encoder** | 매 턴을 기억으로 압축, **매 턴** (싸고 빠른 거). |
| **router** | 작은 의도 분류기 (대화 vs 본작업); 인코더와 맞춰라. |
| **subagent** | 도구 하위작업·병렬 워커 실행; 기본은 chat. |

2단계 표대로 **chat**과 **encoder**를 설정하세요 (router는 encoder와 동일하게).
사용자에게 *왜* 둘인지 설명하세요: *"똑똑한 모델은 당신과 대화해요. 두 번째 싼 모델이
매 턴 끝에 요점을 조용히 적어둬서, 똑똑한 모델이 대화 전체를 다시 안 읽어도 되게
해주죠 — 긴 대화에서도 jarvis가 빠르고 저렴한 비결이에요."*

> **키/모델 직접 설정 (자동 대체 경로).** 대화형 메뉴를 못 다루겠으면
> `~/.jarvis-code/` 아래 파일 두 개를 편집할 수 있다:
> - **모델** → `config.yaml`, 형식은 `프로바이더id/모델id`:
>   ```yaml
>   roles:
>     chat:     anthropic-agent-sdk/claude-opus-4-8
>     subagent: anthropic-agent-sdk/claude-opus-4-8
>     encoder:  anthropic-agent-sdk/claude-haiku-4-5-20251001
>     router:   anthropic-agent-sdk/claude-haiku-4-5-20251001
>   ```
>   (GPT 예: `openai-codex/gpt-5.5` + `openai-codex/gpt-5.4-mini`. Ollama Cloud
>   인코더: `ollama-cloud/devstral-small-2-24b-cloud`.) 실시간 셀렉터가 보여주는
>   정확한 id를 쓸 것.
> - **API 키** → `credentials.yaml`:
>   ```yaml
>   env:
>     BRAVE_SEARCH_API_KEY: "키값"
>     NVIDIA_API_KEY: "nvapi-..."
>   ```

---

## 4단계 — 웹 검색 (Brave) — 무료, 꼭 받아라

**이건 기본 설치 과정으로 챙겨라 — 진짜 왠만하면 받게 밀어붙여라.** 웹 검색 못 하는
코딩 에이전트는 심각하게 불리하다: 최신 문서·패키지 버전·에러 메시지·API를 확인 못
한다. Brave Search API는 **무료 티어 — 월 약 1,000회 검색 무료**, 일반 코딩엔 차고
넘친다.

한 가지 솔직히 짚어라: **무료 플랜도 활성화하려면 신용/체크 카드 등록이 필요하다**
(무료 월 할당량 안에선 청구 안 된다). 그게 유일한 마찰이다. 그래도 사용자에게 강하게
권해라 — 코딩하면서 웹 검색은 진짜 값어치가 있다. 카드 등록이 꺼려지면 존중하고: 그
사실만 알리고 계속 진행해라 (나머지는 다 동작하고, 키는 나중에 추가 가능).

사용자에게 알려주고, 같이 진행하세요:

1. **Brave Search API** 포털로: <https://brave.com/search/api/>
2. 가입 후 **무료 플랜(Free)**을 고르고(활성화에 카드를 요구한다 — 월 할당량 안에선
   무료) API 키를 발급. (페이지가 바뀌었으면 "Brave Search API free tier"로 검색.)
3. 키를 `BRAVE_SEARCH_API_KEY`로 jarvis-code에 등록 — `jarvis api-key`로, 또는
   `~/.jarvis-code/credentials.yaml`에 추가 (3b 참고).

> 왜: *"jarvis가 코딩하면서 실시간 웹을 찾아볼 수 있게 해줘요 — 최신 버전, 진짜 문서,
> 지금 겪는 그 에러요. 무료 티어로 일반 사용은 다 커버되고, 활성화에 카드만 등록하면
> 돼요(할당량 안에선 청구 안 돼요)."*

---

## 5단계 — 이미지 생성 (NVIDIA NIM) — 무료, 받을 값어치 있다

**이것도 설정해라.** 무료이고 jarvis가 할 수 있는 일이 눈에 띄게 늘어난다 — 이미지를
생성·편집할 수 있다(아이콘, 목업, 다이어그램, 에셋). 세상에서 제일 센 이미지 모델은
아니지만, 무료이고 없는 것보단 훨씬 낫다.

1. **NVIDIA build 포털**로: <https://build.nvidia.com/>
2. 로그인 후 API 키 발급 (무료 크레딧으로 시작하기 충분). 키는 `nvapi-...` 형태.
3. 키를 `NVIDIA_API_KEY`로 등록 — `jarvis api-key` → *"NVIDIA NIM (image generation)"*,
   또는 `credentials.yaml`에 추가 (3b).

이러면 `generate_image`, `edit_image` 도구가 켜진다. (이미지 생성은 고정된 FLUX
기본값을 쓴다. `model-setting`에서 이미지 모델은 안 고른다.)

> 왜: *"jarvis가 무료로 이미지를 만들고 고칠 수 있게 돼요 — 만들면서 빠르게 에셋,
> 아이콘, 목업 뽑을 때 유용해요."*

---

## 6단계 — 검증 (건너뛰지 마라 — 이게 됐다는 증거다)

먼저 설치 점검을 실행하세요. `--skip-sidecar`를 써라 — 설치기가 끝나며 사이드카를
멈추므로, 그냥 `jarvis doctor`를 치면 사이드카가 꺼져 있다고 warn이 뜬다(정상, 실패
아님):

```bash
jarvis doctor --skip-sidecar
```

Python, Node, 임베딩 모델, 프로바이더 카탈로그, 인증을 점검합니다. **이게 설치가
멀쩡하다는 1차 증거다.**

그다음 jarvis를 실행 — 에이전트와 백그라운드 사이드카가 같이 뜬다:

```bash
jarvis
```

사이드카가 살아있는지 확인하려면 **다른** 터미널을 열어 조회:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/status | ConvertTo-Json -Depth 6
```

(사이드카는 기본 **8765** 포트. `GET /health`는 `{"ok": true}`를 반환.)

**사용자에게 정직하게 보고하라:** 뭐가 설치됐는지, chat + encoder 모델을 뭘로
설정했는지, 웹 검색·이미지 키가 설정됐는지, `jarvis doctor` 결과는 어떤지. 뭔가
실패했으면 실패했다고, 다음에 뭘 해볼지 말하라 — 검증 안 한 성공을 보고하지 마라.

---

## 뭔가 잘못되면

- 임베딩 모델이 안 받아졌다 → `jarvis doctor --preload-embedder`.
- 프로바이더가 "unavailable, retry"로 뜬다 → 실시간 `/models` 조회 실패다. 키와
  네트워크를 확인하고 `jarvis model-setting`을 다시 열어라.
- 일반 진단 → 우선 `jarvis doctor`. 더 깊은 내용은 GitHub의 `troubleshooting` 문서.

여섯 단계가 다 끝나고 `jarvis doctor`가 통과하면 설치 완료다.
