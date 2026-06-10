# Providers

JARVIS Code has a bundled provider catalog and an optional user overlay:

- Bundled catalog: `scripts/llm_catalog.yaml`
- User overlay: `~/.jarvis-code/llm_catalog.user.yaml`

The overlay is read by both `/model-setting` in the terminal UI and
`scripts/llmsetting.py`. It is only catalog metadata. After you choose chat and
encoder models, JARVIS writes the active runtime routing to
`~/.jarvis-code/config.yaml` and `~/.jarvis-code/providers.yaml`.

## Support Tiers

| Tier | Providers | Notes |
| --- | --- | --- |
| ✅ Tested and bundled | OpenAI OAuth, OpenAI API key, Anthropic, Google Gemini, DashScope, Ollama Cloud | Included in `scripts/llm_catalog.yaml`; use `jarvis gpt-login` for OAuth or `jarvis api-key` for API-key providers. |
| 🟡 Custom OpenAI-compatible | GLM/Zhipu, Kimi/Moonshot, DeepSeek, Groq, OpenRouter, local Ollama, LM Studio, vLLM | Add entries to `~/.jarvis-code/llm_catalog.user.yaml` with `api_format: openai-completions`. |
| ⚪ Custom adapter work | Other wire formats or providers without OpenAI-compatible chat/model APIs | Not selectable from the overlay unless their `api_format` is one of JARVIS Code's known formats. |

Known `api_format` values are `openai-completions`, `anthropic`,
`google-generative-ai`, and `openai-codex-responses`. Custom OpenAI-compatible
providers should use `openai-completions`.

## Role Guide

Use the chat role for your main coding model. For chat, pick the model that best
matches your budget, latency, and reasoning needs.

Use the encoder role for JLC memory encoding. Prefer tested providers and stable
instruction-following models from the bundled catalog. Do not use reasoning
models as the encoder; the encoder should be predictable, inexpensive, and
quiet, not a long-reasoning agent.

## Overlay Format

Create `~/.jarvis-code/llm_catalog.user.yaml`:

```yaml
providers:
  my-openai-compatible:
    label: "My OpenAI-compatible endpoint"
    auth_env: MY_PROVIDER_API_KEY
    base_url: "https://api.example.com/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - my-chat-model
      - my-coder-model
```

Required fields for each custom provider:

- `base_url`
- `api_format`

Useful optional fields:

- `label`: display name in model-setting
- `auth_env`: environment variable that holds the API key
- `models_endpoint`: usually `/models`
- `models_static`: fallback list when the provider has no usable models endpoint

If the overlay YAML is broken, JARVIS prints a warning and uses the bundled
catalog. If one provider entry is invalid, that entry is skipped and the rest of
the catalog still loads.

## Full Example

This example combines common OpenAI-compatible endpoints. Keep only the
providers you use, and set the matching environment variables to `YOUR_KEY_HERE`
in your shell or system environment.

```yaml
providers:
  zhipu:
    label: "GLM / Zhipu"
    auth_env: ZHIPU_API_KEY
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - glm-4.5
      - glm-4.5-air

  moonshot:
    label: "Kimi / Moonshot"
    auth_env: MOONSHOT_API_KEY
    base_url: "https://api.moonshot.ai/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - kimi-k2-0711-preview

  deepseek:
    label: "DeepSeek"
    auth_env: DEEPSEEK_API_KEY
    base_url: "https://api.deepseek.com/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - deepseek-chat
      - deepseek-reasoner

  groq:
    label: "Groq"
    auth_env: GROQ_API_KEY
    base_url: "https://api.groq.com/openai/v1"
    api_format: openai-completions
    models_endpoint: "/models"

  openrouter:
    label: "OpenRouter"
    auth_env: OPENROUTER_API_KEY
    base_url: "https://openrouter.ai/api/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - openai/gpt-oss-120b
      - moonshotai/kimi-k2
      - deepseek/deepseek-chat

  ollama-local:
    label: "Ollama local"
    auth_env: OLLAMA_LOCAL_API_KEY
    base_url: "http://127.0.0.1:11434/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - llama3.1
      - qwen2.5-coder

  lm-studio:
    label: "LM Studio"
    auth_env: LM_STUDIO_API_KEY
    base_url: "http://127.0.0.1:1234/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - local-model

  vllm:
    label: "vLLM"
    auth_env: VLLM_API_KEY
    base_url: "http://127.0.0.1:8000/v1"
    api_format: openai-completions
    models_endpoint: "/models"
    models_static:
      - served-model-name
```

PowerShell example:

```powershell
$env:DEEPSEEK_API_KEY = "YOUR_KEY_HERE"
jarvis model-setting
```

Bash example:

```bash
export DEEPSEEK_API_KEY="YOUR_KEY_HERE"
jarvis model-setting
```
