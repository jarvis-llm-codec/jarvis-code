# Providers

JARVIS Code has a bundled provider catalog and an optional user overlay:

- Bundled catalog: `scripts/llm_catalog.yaml`
- User overlay: `~/.jarvis-code/llm_catalog.user.yaml`

Use `/api-key` as the main provider setup entrypoint. It can save keys for
bundled providers, add OpenAI-compatible custom providers, change custom keys,
and remove custom providers. After a provider is configured, use
`/model-setting` to pick chat and encoder models.

The overlay is read by both `/api-key`, `/model-setting` in the terminal UI, and
`scripts/llmsetting.py`. It is only catalog metadata. After you choose chat and
encoder models, JARVIS writes the active runtime routing to
`~/.jarvis-code/config.yaml` and `~/.jarvis-code/providers.yaml`.

## Support Tiers

| Tier | Providers | Notes |
| --- | --- | --- |
| ✅ Tested and bundled | OpenAI OAuth, OpenAI API key, Anthropic, Google Gemini, DashScope, Ollama Cloud, OpenRouter | Included in `scripts/llm_catalog.yaml`; use `jarvis gpt-login` for OAuth or `jarvis api-key` for API-key providers. Model lists are fetched live (`:free` tiers group to the top on OpenRouter). |
| ✅ Local, bundled, keyless | Ollama (`localhost:11434`), LM Studio (`localhost:1234`), llama.cpp server (`localhost:8080`) | No key, no setup — start the local server and your models appear in `/model-setting`. A stopped server just shows as unavailable. Different port? Override `base_url` in the user overlay. |
| ✅ Image generation | NVIDIA NIM | Add the NVIDIA key with `/api-key`; it enables `generate_image` and `edit_image`. |
| 🟡 Custom OpenAI-compatible | GLM/Zhipu, Kimi/Moonshot, DeepSeek, Groq, vLLM, anything else speaking the OpenAI API | Use `/api-key` → `Add custom provider...`; JARVIS writes `~/.jarvis-code/llm_catalog.user.yaml` for you. Leave the key empty for keyless local endpoints. |
| ⚪ Custom adapter work | Other wire formats or providers without OpenAI-compatible chat/model APIs | Not selectable from the overlay unless their `api_format` is one of JARVIS Code's known formats. |

Known `api_format` values are `openai-completions`, `anthropic`,
`google-generative-ai`, and `openai-codex-responses`. Custom OpenAI-compatible
providers should use `openai-completions`.

## Role Guide

Use the chat role for your main coding model. For chat, pick the model that best
matches your budget, latency, and reasoning needs.

The chat model is the driver: JARVIS Code is a harness, and building means
sustained tool calling — register a project, write files, verify, repeat — under
a large prompt. Small or free-tier models often chat fine but stall at acting:
they announce a plan and end the turn without calling tools, or manage one tool
call and stop. That is a model limitation, not a configuration problem. For
real build work, use a model with strong agentic tool calling (GPT, Claude,
GLM, or Qwen3-coder class). Free aggregator tiers (e.g. OpenRouter `:free`) are
shared capacity — expect upstream rate limits — and are best treated as chat
and experimentation tiers rather than build drivers.

Use the encoder role for JLC memory encoding. Prefer tested providers and stable
instruction-following models from the bundled catalog. Do not use reasoning
models as the encoder; the encoder should be predictable, inexpensive, and
quiet, not a long-reasoning agent.

## Image Generation

Add your NVIDIA key through `/api-key` -> `NVIDIA NIM (image generation)`.
That enables `generate_image` and `edit_image`. Image generation stays on the
fixed FLUX defaults; `/model-setting` only chooses chat and encoder models.

## Add A Custom Provider

In the terminal UI:

```text
/api-key
```

Choose `Add custom provider...`, then enter:

- base URL, such as `https://api.example.com/v1`
- display name, such as `DeepSeek` or `GLM / Zhipu`
- API key

JARVIS derives the provider id from the label, stores the key in the normal
credentials file, writes the catalog entry to
`~/.jarvis-code/llm_catalog.user.yaml`, and immediately tries the provider's
`/models` endpoint. If the fetch fails, the provider is still saved; fix the URL
or key and run `/api-key` again to change the key.

Then run:

```text
/model-setting
```

For shell setup without the terminal UI, use:

```bash
jarvis api-key
jarvis model-setting
```

## Advanced Overlay Format

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

## Advanced Full Example

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

  # OpenRouter, Ollama (local), LM Studio, and llama.cpp server are bundled
  # presets now — no overlay entry needed. To change a bundled preset's
  # endpoint (e.g. llama.cpp on a non-default port), override just that field;
  # the overlay deep-merges over the bundled entry:
  llamacpp:
    base_url: "http://127.0.0.1:8081/v1"

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
