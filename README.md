# Qwen Image API for ComfyUI

A small ComfyUI custom node for calling Qwen image edit APIs through either:

- OpenAI-compatible image edit gateways, such as `/v1/images/edits`
- DashScope multimodal generation API

The node accepts one to three ComfyUI `IMAGE` inputs and returns a normal ComfyUI `IMAGE`, so it can connect directly to `SaveImage`, `PreviewImage`, upscalers, or other image nodes.

## Install

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone <REPOSITORY_URL> qwen_image_api
```

Then restart ComfyUI.

## Node

After restart, add:

```text
api/qwen/Qwen Image Edit API
```

## Key Settings

For an OpenAI-compatible gateway:

```text
api_url = https://ai-gw.emdlz.com.cn/
request_style = openai_images_edit_multipart
model = qwen-image-edit-plus
```

For the API key field:

```text
api_key_or_env = QWEN_API_KEY
```

or directly:

```text
api_key_or_env = sk-...
```

Direct keys are saved as plain text inside workflow JSON files. If you share a workflow, remove the key or change it back to an environment variable name first.

## Environment Variable Example

Windows PowerShell:

```powershell
setx QWEN_API_KEY "sk-your-key"
```

Restart ComfyUI after setting the environment variable.

## Notes

- `api_url` values ending in `/v1` or `/v1/images/edits` are supported.
- Large ComfyUI seed values are accepted and normalized before API calls.
- Duplicate URLs in gateway responses are filtered to avoid saving the same image twice.
