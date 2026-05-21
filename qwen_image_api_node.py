import base64
import io
import os
import re
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import torch
from PIL import Image


DASHSCOPE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def _first_image_tensor(image):
    if image is None:
        return None
    if len(image.shape) == 4:
        return image[0]
    return image


def _tensor_to_pil(image):
    img = _first_image_tensor(image)
    if img is None:
        return None
    arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    mode = "RGBA" if len(arr.shape) == 3 and arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode=mode)


def _pil_to_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _image_to_data_url(image):
    png = _pil_to_png_bytes(image)
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pil_to_tensor(image):
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _normalize_dashscope_url(api_url):
    api_url = (api_url or "").strip()
    if not api_url:
        api_url = "https://dashscope.aliyuncs.com"

    parsed = urlparse(api_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/generation"):
        return api_url
    if path.endswith("/api/v1"):
        return api_url.rstrip("/") + "/services/aigc/multimodal-generation/generation"
    if not path:
        return api_url.rstrip("/") + DASHSCOPE_PATH
    return api_url


def _normalize_openai_edit_url(api_url):
    api_url = (api_url or "").strip()
    if not api_url:
        raise ValueError("api_url is empty")
    parsed = urlparse(api_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/images/edits"):
        return api_url
    if path.endswith("/v1"):
        return api_url.rstrip("/") + "/images/edits"
    if not path:
        return api_url.rstrip("/") + "/v1/images/edits"
    return api_url


def _force_openai_style(api_url):
    parsed = urlparse((api_url or "").strip())
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/").lower()
    return host == "ai-gw.emdlz.com.cn" or path in {"/v1", "/v1/images/edits"}


def _format_size(width, height, request_style):
    if request_style == "openai_images_edit_multipart":
        return f"{width}x{height}"
    return f"{width}*{height}"


def _round_size(value):
    return max(64, int(round(float(value) / 16) * 16))


def _size_from_ratio(request_style, aspect_ratio, long_edge, output_width, output_height):
    aspect_ratio = (aspect_ratio or "auto_from_input").strip()
    long_edge = max(64, int(long_edge or 1024))

    ratios = {
        "1:1 square": (1, 1),
        "3:4 portrait": (3, 4),
        "4:3 landscape": (4, 3),
        "9:16 portrait": (9, 16),
        "16:9 landscape": (16, 9),
        "2:3 portrait": (2, 3),
        "3:2 landscape": (3, 2),
    }

    if aspect_ratio == "custom_width_height":
        width = _round_size(output_width)
        height = _round_size(output_height)
        return _format_size(width, height, request_style)

    if aspect_ratio in ratios:
        ratio_w, ratio_h = ratios[aspect_ratio]
        if ratio_w >= ratio_h:
            width = _round_size(long_edge)
            height = _round_size(long_edge * ratio_h / ratio_w)
        else:
            height = _round_size(long_edge)
            width = _round_size(long_edge * ratio_w / ratio_h)
        return _format_size(width, height, request_style)

    return None


def _size_from_image(image, request_style, size, aspect_ratio="auto_from_input", long_edge=1024, output_width=1024, output_height=1024):
    ratio_size = _size_from_ratio(request_style, aspect_ratio, long_edge, output_width, output_height)
    if ratio_size:
        return ratio_size

    size = (size or "").strip()
    if size and size.lower() not in {"auto", "auto_from_input"}:
        return size

    pil = _tensor_to_pil(image)
    if pil is None:
        return ""
    width, height = pil.size
    if width <= 0 or height <= 0:
        return ""

    scale = min(2048 / width, 2048 / height, 1.0)
    width = max(512, int(round(width * scale / 16) * 16))
    height = max(512, int(round(height * scale / 16) * 16))

    return _format_size(width, height, request_style)


def _extract_urls_or_b64(data):
    found = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and not item.strip():
                    continue
                key_l = str(key).lower()
                if key_l in {"url", "image", "image_url"} and isinstance(item, str):
                    found.append(("url", item))
                elif key_l in {"b64_json", "base64", "image_base64"} and isinstance(item, str):
                    found.append(("b64", item))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return found


def _normalize_seed(seed):
    seed = int(seed or 0)
    if seed <= 0:
        return 0
    return seed % 2147483647


def _image_from_b64(text):
    if "," in text and text.strip().startswith("data:"):
        text = text.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(text))).convert("RGB")


class QwenImageEditAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": "qwen-image-edit-plus"}),
                "api_url": ("STRING", {"default": "https://ai-gw.emdlz.com.cn/"}),
                "api_key_or_env": ("STRING", {"default": "QWEN_API_KEY"}),
                "request_style": (
                    ["dashscope_multimodal_json", "openai_images_edit_multipart"],
                    {"default": "openai_images_edit_multipart"},
                ),
                "size": ("STRING", {"default": "auto_from_input"}),
                "aspect_ratio": (
                    [
                        "auto_from_input",
                        "1:1 square",
                        "3:4 portrait",
                        "4:3 landscape",
                        "9:16 portrait",
                        "16:9 landscape",
                        "2:3 portrait",
                        "3:2 landscape",
                        "custom_width_height",
                    ],
                    {"default": "auto_from_input"},
                ),
                "output_long_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "output_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "output_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 1800}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "image_urls")
    FUNCTION = "generate"
    CATEGORY = "api/qwen"

    def _api_key(self, api_key_or_env):
        value = (api_key_or_env or "").strip()
        if value.startswith(("sk-", "dashscope-")):
            return value

        names = [x.strip() for x in re.split(r"[,; ]+", value) if x.strip()]
        names.extend(["QWEN_API_KEY", "DASHSCOPE_API_KEY", "ALIYUN_API_KEY"])
        for name in names:
            env_value = os.environ.get(name)
            if env_value:
                return env_value
        raise RuntimeError(
            "Missing API key. Put sk-... directly in api_key_or_env, or set QWEN_API_KEY and restart ComfyUI."
        )

    def _post_dashscope(
        self,
        api_url,
        api_key,
        model,
        prompt,
        negative_prompt,
        size,
        seed,
        n,
        prompt_extend,
        watermark,
        timeout,
        images,
    ):
        content = []
        for image in images:
            if image is not None:
                content.append({"image": _image_to_data_url(image)})
        content.append({"text": prompt})

        parameters = {
            "n": n,
            "prompt_extend": bool(prompt_extend),
            "watermark": bool(watermark),
        }
        if size:
            parameters["size"] = size
        if seed:
            parameters["seed"] = seed
        if negative_prompt.strip():
            parameters["negative_prompt"] = negative_prompt

        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": parameters,
        }

        response = requests.post(
            _normalize_dashscope_url(api_url),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Qwen API error {response.status_code}: {response.text[:1000]}")
        return response.json()

    def _post_openai_edit(
        self,
        api_url,
        api_key,
        model,
        prompt,
        negative_prompt,
        size,
        seed,
        n,
        timeout,
        images,
    ):
        files = []
        first = True
        for image in images:
            if image is None:
                continue
            field = "image" if first else "image[]"
            files.append((field, ("image.png", _pil_to_png_bytes(image), "image/png")))
            first = False
        if not files:
            raise RuntimeError("openai_images_edit_multipart requires at least one image.")

        data = {"model": model, "prompt": prompt, "n": str(n)}
        if negative_prompt.strip():
            data["negative_prompt"] = negative_prompt
        if size:
            data["size"] = size.replace("*", "x")
        if seed:
            data["seed"] = str(seed)

        response = requests.post(
            _normalize_openai_edit_url(api_url),
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files=files,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Qwen API error {response.status_code}: {response.text[:1000]}")
        return response.json()

    def _download_images(self, data, api_url, timeout):
        images = []
        urls = []
        seen = set()
        for kind, value in _extract_urls_or_b64(data):
            marker = (kind, value)
            if marker in seen:
                continue
            seen.add(marker)
            if kind == "b64":
                images.append(_image_from_b64(value))
                urls.append("base64")
                continue

            if value.startswith("data:"):
                images.append(_image_from_b64(value))
                urls.append("base64")
                continue

            url = value if bool(urlparse(value).scheme) else urljoin(api_url, value)
            img_response = requests.get(url, timeout=timeout)
            if img_response.status_code >= 400:
                raise RuntimeError(f"Failed to download generated image {img_response.status_code}: {url}")
            images.append(Image.open(io.BytesIO(img_response.content)).convert("RGB"))
            urls.append(url)

        if not images:
            raise RuntimeError(f"Qwen API returned no image URL/base64. Response: {str(data)[:1000]}")
        return images, urls

    def generate(
        self,
        image1,
        prompt,
        negative_prompt,
        model,
        api_url,
        api_key_or_env,
        request_style,
        size,
        aspect_ratio,
        output_long_edge,
        output_width,
        output_height,
        seed,
        n,
        prompt_extend,
        watermark,
        timeout,
        image2=None,
        image3=None,
    ):
        api_key = self._api_key(api_key_or_env)
        if request_style != "openai_images_edit_multipart" and _force_openai_style(api_url):
            request_style = "openai_images_edit_multipart"
        pil_images = [_tensor_to_pil(image1), _tensor_to_pil(image2), _tensor_to_pil(image3)]
        resolved_size = _size_from_image(
            image1,
            request_style,
            size,
            aspect_ratio,
            output_long_edge,
            output_width,
            output_height,
        )
        seed = _normalize_seed(seed)

        if request_style == "openai_images_edit_multipart":
            data = self._post_openai_edit(
                api_url,
                api_key,
                model,
                prompt,
                negative_prompt,
                resolved_size,
                seed,
                n,
                timeout,
                pil_images,
            )
        else:
            data = self._post_dashscope(
                api_url,
                api_key,
                model,
                prompt,
                negative_prompt,
                resolved_size,
                seed,
                n,
                prompt_extend,
                watermark,
                timeout,
                pil_images,
            )

        out_images, urls = self._download_images(data, api_url, timeout)
        tensors = [_pil_to_tensor(img) for img in out_images]
        if len(tensors) == 1:
            return (tensors[0], "\n".join(urls))
        return (torch.cat(tensors, dim=0), "\n".join(urls))


NODE_CLASS_MAPPINGS = {
    "QwenImageEditAPI": QwenImageEditAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditAPI": "Qwen Image Edit API",
}
