# jarvis-code 설치 — 당신의 AI를 위한 가이드

**이 페이지는 AI 어시스턴트를 위해 쓰였습니다.** 사람이라면: 이 단계를 직접 따라하지
마세요. 아래 한 줄을, 당신 컴퓨터에서 터미널 명령을 실행할 수 있는 AI — **Claude Code,
Codex, Cursor, Windsurf** 등 — 에게 붙여넣으세요:

> *jarvis-code를 대신 설치해줘. 먼저 이 가이드 전체를 읽고 정확히 그대로 따라가:
> https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/docs/ai-install.md
> — 내가 어떤 AI를 이미 쓰는지 물어보고, 거기에 맞춰 세팅하고, 무료 API 키까지 같이
> 설정해줘.*

AI가 이 가이드를 가져와 작업을 대신 해줍니다. 당신은 질문 몇 개에 답하고 무료 가입
두 개를 클릭하면 됩니다.

---

## 이 글을 읽는 AI에게

당신은 이 사용자의 컴퓨터에 **jarvis-code**를 설치·설정하고 있습니다. **아무것도 하기
전에 이 페이지 전체를 읽고**, 짧은 계획을 세운 뒤 따라가세요.

당신의 일은 명령어 실행만이 아닙니다:

1. 앱을 **설치**한다.
2. **사용자가 이미 가진 AI가 뭔지 파악**하고(구독 / API 키), 거기에 맞춰 jarvis-code를
   최적으로 설정한다 — 두 번째 프로바이더에 돈을 쓰게 하지 마라.
3. **각 단계가 왜 중요한지 쉬운 말로 설명**한다 — 특히 *인코더*. 사용자는 이걸 거의
   확실히 들어본 적이 없고, 당신이 설명해주지 않으면 이해하지 못한다.
4. **무료 API 키 두 개**(웹 검색 + 이미지 생성)를 설정한다. 일반 사용 범위에선 무료이고
   차이가 크니, 일반 설치의 일부처럼 취급하라. 강하게 권하되, 사용자가 거절하거나 가입이
   막히면 기록하고 **계속 진행하라: 기본 설치는 이것들 없이도 동작하고**, 키는 나중에
   추가할 수 있다.
5. 성공을 선언하기 전에, **두 모델 역할(채팅 *그리고* 인코더)이 모두 설정됐는지** 포함해
   실제로 동작하는지 **검증**한다.

행동 원칙: 사용자는 기술자가 아닐 수 있다. 어떤 단계는 사람이 해야 한다(브라우저 로그인,
웹사이트 가입, 메뉴 선택) — 그런 건 *정확히* 뭘 어떻게 하라고 알려주고 기다려라. Step 6의
검증을 돌리고 실제 결과를 보고하기 전엔 절대 "완료"라고 하지 마라.

### jarvis-code가 뭔지 (당신이 설치하는 게 뭔지 이해하도록)

jarvis-code는 하나의 아이디어를 중심으로 만든 터미널 코딩 에이전트입니다: **채팅 모델은
대화 기록을 절대 받지 않는다.** 매 턴, 모델은 작은 이월 메모리 문서("JHB")와 새 메시지만
받습니다 — 세션이 아무리 길어져도 입력이 거의 *일정하게* 유지되며, 매 턴 늘어나지 않습니다.

이걸 가능하게 하는 게 두 번째의 값싼 모델 — **인코더**입니다: 매 턴이 끝나면 방금 일어난
일을 그 작은 메모리 문서로 압축해, 다음 턴이 *대화록*을 물려받지 않고도 *의미*를 물려받게
합니다. 그래서 jarvis-code는 모델을 *두 개*(채팅 + 인코더) 설정하라고 요구하며, 이게
사용자가 직관적으로 이해 못 하는 부분입니다. Step 3에서 쓸 수 있게 이 설명을 준비해 두세요.

---

## Step 1 — 설치

사전 요구사항(Windows에서는 설치기가 `winget`으로 없는 것들을 자동 설치):

- **Node.js 20+**, **Python 3.10+**, **Git**
- **Microsoft Visual C++ 2015–2022 재배포 가능 패키지 (x64)** (Windows)
- 최초 설치 시 인터넷 접속

오늘 지원·검증된 플랫폼은 Windows입니다(macOS / Linux는 준비 중).

**PowerShell**에서 공식 설치기를 실행:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

하는 일: 엔진용 Python venv를 만들고, Node 의존성을 설치하고, 메모리 회상에 쓰는
**`BAAI/bge-m3` 임베딩 모델(디스크 ~4.3 GB)을 내려받습니다.** 총 용량은
`%LOCALAPPDATA%\JARVIS-Code` 아래 **≈6 GB**입니다. 다운로드가 느린 부분이니 정상이고
한 번뿐이라고 사용자에게 알려주세요.

임베딩 모델 다운로드가 실패해도 설치는 중단되지 않습니다. 나중에 다시:

```powershell
jarvis doctor --preload-embedder
```

> 사용자에게 당신 말로: *"~6 GB짜리 툴킷을 설치하는데, 대부분은 jarvis가 매번 다시 읽지
> 않고도 기억을 떠올릴 수 있게 하는 로컬 검색 모델이에요. 첫 설치는 좀 걸리고, 그 뒤엔
> 빨라요."*

**설치 후, 새 터미널을 여세요.** 설치기가 `jarvis` 명령을 PATH에 넣지만, *이* 터미널은
다시 열기 전까지 인식하지 못합니다(설치기 자체도 "이 창을 닫고 새 터미널을 열라"고
출력합니다). 아래 단계에서 `jarvis ...` 명령을 실행하기 전에 새 PowerShell 창을 여세요.

---

## Step 2 — 사용자 인터뷰 (여기서 모든 게 결정됨)

무엇을 설정하기 전에, 사용자가 **이미 뭘 쓰는지** 물어보세요. 이게 어떤 모델을 고를지,
API 키가 필요한지를 결정합니다:

> *"지금 어떤 AI에 돈을 내거나 쓰고 계세요? 예: ChatGPT/OpenAI 구독, Claude(Anthropic)
> 구독, 알리바바 DashScope의 GLM / Qwen, Ollama Cloud, 아니면 다른 것? 여러 개여도
> 괜찮습니다."*

**기본 — GPT.** 가장 흔한 경우는 압도적으로 **ChatGPT / OpenAI 구독**이며, 새 설치는 이미
그 조합을 기본으로 출하됩니다: 채팅 `openai-codex/gpt-5.5` + 인코더
`openai-codex/gpt-5.4-mini`. 그러니 GPT라면 대체로 기본값을 확인하는 것뿐입니다. 다른
걸 쓴다면, 아래 표를 보고 **두 역할 모두** 거기에 맞게 바꾸세요.

아래 표(플랜별 권장 조합)로 답을 매핑하세요:

모델 id는 `provider-id/model-id` 형식으로 표기합니다 — `config.yaml`(Step 3b)에서 쓰는
정확한 형식입니다. 메뉴로 설정할 땐 프로바이더와 모델을 따로 고릅니다.

| 사용자가 이미 가진 것 | 채팅 + 서브에이전트 | 인코더 (값싸고, 매 턴) |
|---|---|---|
| **OpenAI / GPT** (구독) — **기본** | `openai-codex/gpt-5.5` | `openai-codex/gpt-5.4-mini` |
| **Claude (Anthropic)** (구독) | `anthropic-agent-sdk/claude-opus-4-8` | `anthropic-agent-sdk/claude-haiku-4-5-20251001` |
| **Ollama Cloud** | `ollama-cloud/glm-5-ollama` (또는 다른 프런티어 모델) | `ollama-cloud/devstral-small-2-24b-cloud` |
| **DashScope (GLM / Qwen)** | `dashscope/glm-5` | `dashscope/…` ~14–24B 모델 |

인코더 기준: 그들의 플랜에서 **가장 싸면서 쓸 만한, 대략 14–24B급** 모델을 고르세요.
**8B는 너무 작습니다**(메모리가 불안정해짐). **추론 모델을 인코더로 쓰지 마세요** — 빠르고
싸고 예측 가능해야 합니다. 매 턴 돌기 때문에, 무거운 모델을 여기 쓰면 이득 없이 돈이나
레이트리밋만 태웁니다.

사용자가 **유료로 가진 게 없어도** 괜찮습니다 — **Ollama Cloud**(가입하면 무료 키)나
Ollama / LM Studio를 통한 로컬 모델을 쓸 수 있다고 알려주고, 가장 작은 합리적인 조합을
고르세요. (이걸로 설치를 막지 마세요.)

---

## Step 3 — 채팅 프로바이더 인증 후, 모델 설정

### 3a. 인증 (사람이 하는 단계 하나)

사용자의 플랜에 맞는 경로를 고르세요. **Claude 경로는 특별합니다** — 꼭 읽으세요.

- **OpenAI / ChatGPT 구독 (기본) → OAuth 로그인.** 사용자에게 터미널에서 실행하게 하세요:
  ```bash
  jarvis gpt-login
  ```
  브라우저가 열립니다. 브라우저로 완료가 안 되면 `jarvis gpt-login-device`를 쓰세요.
- **Claude (Anthropic) 구독 → API 키 불필요.** jarvis-code는 `anthropic-agent-sdk`
  프로바이더를 통해 사용자의 **Claude Code** OAuth를 재사용합니다 — 구독으로 청구되고
  별도 키가 없습니다. 실행:
  ```bash
  jarvis claude-login
  ```
  Claude Code의 `setup-token` 플로우를 돌려 토큰을 jarvis-code용으로 캡처합니다.
  `claude` 명령이 없으면 `npx @anthropic-ai/claude-code`로 폴백할 수 있습니다(강제하려면
  `jarvis claude-login --npx`). 이 경로의 인코더는 일부러 Haiku급으로 제한됩니다 — 매 턴
  도는 인코더가 구독 레이트리밋을 갉아먹지 않도록.
- **API 키 프로바이더 전반**(OpenAI API, Anthropic API, Gemini, DashScope, Ollama Cloud,
  OpenRouter): `jarvis api-key`를 실행해 프롬프트를 따르거나, 키를 직접 씁니다(아래 참고).
  키 환경변수:
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DASHSCOPE_CODING_API_KEY`,
  `OLLAMA_API_KEY`, `OPENROUTER_API_KEY`.

### 3b. 모델 설정 — 두 역할 모두 (설치가 어긋나는 지점)

> ⚠️ **가장 흔한 실수: 채팅만 설정하고 인코더는 안 하는 것.** 로그인(예: `jarvis gpt-login`)
> 하고 채팅 모델을 고른다고 **인코더가 설정되진 않습니다** — 이전 값 그대로 남습니다.
> **인코더 역할을 반드시 명시적으로 설정**해야 합니다. 안 그러면 사용자는 채팅 모델은 맞고
> 인코더는 어긋난 상태가 됩니다. **둘 다** 설정하고, Step 6에서 **둘 다 검증**하세요.

**확실한 방법 — `config.yaml`을 직접 쓰기.** 가장 신뢰할 수 있는 방법은(반쯤 하다 만
메뉴 없이) 전체 역할 세트를 `~/.jarvis-code/config.yaml`에 쓰는 것입니다. 사용자의 플랜에
맞는 블록을 고르세요 — **각 블록이 네 역할 모두를 설정하므로 인코더가 방치될 수 없습니다:**

**OpenAI / GPT 구독 (기본):**
```yaml
roles:
  chat:     openai-codex/gpt-5.5
  subagent: openai-codex/gpt-5.5
  encoder:  openai-codex/gpt-5.4-mini
  router:   openai-codex/gpt-5.4-mini
```

**Claude (Anthropic) 구독:**
```yaml
roles:
  chat:     anthropic-agent-sdk/claude-opus-4-8
  subagent: anthropic-agent-sdk/claude-opus-4-8
  encoder:  anthropic-agent-sdk/claude-haiku-4-5-20251001
  router:   anthropic-agent-sdk/claude-haiku-4-5-20251001
```

**Ollama Cloud:**
```yaml
roles:
  chat:     ollama-cloud/glm-5-ollama
  subagent: ollama-cloud/glm-5-ollama
  encoder:  ollama-cloud/devstral-small-2-24b-cloud
  router:   ollama-cloud/devstral-small-2-24b-cloud
```

(라이브 선택기가 그 프로바이더에 대해 다른 id를 보여주면, 그 정확한 id를 쓰세요.)
API 키는 `~/.jarvis-code/credentials.yaml`에 넣습니다:
```yaml
env:
  BRAVE_SEARCH_API_KEY: "the-key"
  NVIDIA_API_KEY: "nvapi-..."
```

**또는 메뉴를 대화형으로 몰기:**
```bash
jarvis model-setting
```
**라이브** 모델 목록을 가져와 역할을 순서대로 안내합니다 — **채팅, 그다음 인코더, 그다음
라우터** — 그러니 *채팅에서 멈추지 마세요.* 역할:

| 역할 | 하는 일 |
|---|---|
| **chat** | 사용자가 대화하는 모델 — 코딩 & 대화 (똑똑함). |
| **encoder** | 매 턴 그 턴을 메모리로 압축 (값싸고 빠름). |
| **router** | 작은 의도 분류기(채팅 vs 깊은 작업); 인코더를 미러링. |
| **subagent** | 툴 서브태스크와 병렬 워커 실행; 채팅을 미러링. |

사용자에게 *왜* 두 개인지 설명하세요: *"똑똑한 모델이 당신과 대화하고, 두 번째의 값싼
모델이 매 턴 요지를 조용히 받아 적어서 똑똑한 모델이 대화 전체를 다시 읽을 필요가 없게
합니다 — 그게 긴 세션에서 jarvis를 빠르고 저렴하게 유지하는 핵심입니다."*

---

## Step 4 — 웹 검색 (Brave) — 일반 사용엔 무료, 그리고 꼭 필요함

**일반 설치의 일부로 설정하세요 — 밀어붙이세요.** 웹을 검색 못 하는 코딩 에이전트는 크게
불리합니다: 최신 문서, 패키지 버전, 에러 메시지, API를 확인할 수 없습니다.

**비용에 대해 사용자에게 아주 명확하게 말하세요. 오해해서 포기하기 쉽습니다.** Brave의
Search API는 **매월 $5어치 무료 검색 크레딧 — 약 1,000회 검색 — 을 무료로** 줍니다. 일반적인
코딩은 그 한참 아래이므로, **보통 사용자에겐 사실상 무료입니다.** 정직한 단서 두 개, 둘 다
진짜 걸림돌은 아닙니다:

- **활성화에 카드 등록이 필요** — 신원 확인용일 뿐입니다. 월 무료 크레딧 안에 있는 한
  **청구되지 않습니다.**
- 청구는 **월 ~1,000회 검색을 초과할 때만**(거의 그럴 일 없음), 그리고 무료 티어에는
  작은 **출처 표기(attribution)** 를 요청합니다.

> **사용자에게 "$5 든다"고 말하지 마세요.** 그게 사람들이 포기하게 만드는 흔한 오해입니다.
> **$5어치의 *무료* 크레딧** — 즉 일반 사용엔 무료 — 이고, 카드는 청구가 아니라 확인용일
> 뿐입니다. 없는 비용 때문에 포기하지 않도록 분명히 말해주세요.

같이 진행하세요:

1. **Brave Search API** 포털로 이동: <https://brave.com/search/api/>
2. 가입하고, **무료 플랜**(유료 티어가 *아니라* 월 무료 크레딧이 있는 것)을 고르고, 활성화용
   카드를 등록하고(무료 크레딧 안에선 청구 안 됨), API 키를 만듭니다. (레이아웃이 바뀌었으면
   "Brave Search API free plan"으로 검색하세요.)
3. 키를 `BRAVE_SEARCH_API_KEY`로 jarvis-code에 주세요 — `jarvis api-key`로 하거나,
   `~/.jarvis-code/credentials.yaml`에 추가(Step 3b 참고).

> 이렇게 말하세요: *"이걸로 jarvis가 코딩하면서 실시간 웹을 검색해요 — 최신 버전, 진짜
> 문서, 지금 겪는 그 에러까지. 일반 사용엔 무료예요: 한 달 약 1,000회 검색까지 무료이고,
> 카드는 청구가 아니라 확인용일 뿐이에요."*

---

## Step 5 — 이미지 생성 (NVIDIA NIM) — 무료, 할 만함

**이것도 설정하세요.** 무료이고 jarvis가 할 수 있는 일을 눈에 띄게 넓힙니다 — 이미지 생성·편집
(아이콘, 목업, 다이어그램, 에셋). 세상에서 가장 강력한 이미지 모델은 아니지만, 무료이고 없는
것보다 훨씬 낫습니다.

1. **NVIDIA의 build 포털**로 이동: <https://build.nvidia.com/>
2. 로그인하고 API 키를 만듭니다(무료 크레딧이 시작하기에 충분). 키는 `nvapi-...` 형태입니다.
3. 키를 `NVIDIA_API_KEY`로 jarvis-code에 주세요 — `jarvis api-key` →
   *"NVIDIA NIM (image generation)"* 로 하거나, `credentials.yaml`에 추가(Step 3b).

`generate_image`와 `edit_image` 툴이 켜집니다. (이미지 생성은 고정 FLUX 기본값을 쓰며,
`model-setting`에서 이미지 모델을 고르지 않습니다.)

> 이유: *"이걸로 jarvis가 무료로 이미지를 만들고 편집할 수 있어요 — 빌드하면서 빠른 에셋,
> 아이콘, 목업에 편해요."*

---

## Step 6 — 검증 (건너뛰지 마세요 — 이게 동작을 증명하는 방법)

먼저 설치 점검을 실행하세요. `--skip-sidecar`를 쓰세요 — 설치기가 종료 시 사이드카를 멈추므로,
그냥 `jarvis doctor`는 사이드카가 꺼졌다고 경고합니다(정상이지, 실패가 아님):

```bash
jarvis doctor --skip-sidecar
```

Python, Node, 임베딩 모델, 프로바이더 카탈로그, 인증을 점검합니다. **이게 설치가 온전하다는
1차 증거입니다.**

**두 모델 역할이 모두 설정됐는지 검증하세요** — 흔한 인코더 버그가 드러나는 지점입니다.
`~/.jarvis-code/config.yaml`을 열거나(`jarvis model-setting`으로 현재 값을 읽어) **채팅과
인코더 둘 다** 그들의 플랜에 맞는지 확인하세요 — 예를 들어 GPT라면 채팅
`openai-codex/gpt-5.5`, 인코더 `openai-codex/gpt-5.4-mini`. 인코더가 플랜과 안 맞는
기본값에 남아 있으면, 성공 선언 전에 **지금 고치세요.**

그다음 jarvis를 실행 — 에이전트 *그리고* 그 백그라운드 사이드카를 시작합니다:

```bash
jarvis
```

사이드카가 살아 있는지 확인하려면, **다른** 터미널을 열어 찔러보세요:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/status | ConvertTo-Json -Depth 6
```

(사이드카는 기본으로 **8765** 포트에서 듣습니다; `GET /health`는 `{"ok": true}`를 반환.)

**사용자에게 정직하게 보고하세요:** 무엇이 설치됐는지, **어떤 채팅·인코더 모델이 설정됐는지**
(둘 다 이름을 대세요), 웹 검색·이미지 키가 설정됐는지, `jarvis doctor` 결과. 뭔가 실패했으면
그렇다고 말하고 다음에 뭘 시도할지 말하세요 — 검증 안 한 성공을 보고하지 마세요.

---

## 뭔가 잘못되면

- 임베딩 모델이 안 받아졌으면 → `jarvis doctor --preload-embedder`.
- 프로바이더가 "unavailable, retry"를 보이면 → 라이브 `/models` 조회가 실패한 것; 키와
  네트워크를 확인하고 `jarvis model-setting`을 다시 여세요.
- 채팅은 되는데 메모리가 이상하면 → 인코더가 아직 잘못된 모델일 가능성이 큼;
  `config.yaml`의 `roles.encoder`를 다시 확인하세요(Step 3b / Step 6).
- 일반 진단 → `jarvis doctor` 먼저; 더 깊은 노트는 프로젝트의 GitHub `troubleshooting`
  문서에.

여섯 단계가 모두 끝나고, `jarvis doctor`가 통과하고, **채팅과 인코더 둘 다** 사용자의 플랜에
맞게 설정되면, 설치가 완료된 것입니다.
