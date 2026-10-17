"""Small OpenAI-compatible multimodal client used by the nerve pipeline.

The client deliberately keeps provider differences in ``config.yaml``.  It returns
JSON-serialisable records so the caller can save the original response alongside
parsed detections.  API keys are read only at request time and are never included
in a returned record or log message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

try:  # PyYAML is listed in requirements.txt, but a useful fallback helps dry-run.
    import yaml
except Exception:  # pragma: no cover - only used when PyYAML is unavailable
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_KEY_FILE = ROOT.parent / "api" / "JoyBuilder_api.txt"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# No hidden client-side cap: the caller's ``max_requests`` and the provider's
# ``max_attempts`` remain the explicit controls.  The CLI still defaults to 1.
HARD_MAX_REQUESTS = None

# These defaults also make dry-run useful before config.yaml has been created.
DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "jd_gt6": {"provider": "jdcloud", "model": "gt-6-as-a"},
    "jd_qwen": {"provider": "openrouter", "model": "qwen/qwen3.8-omni-flash"},
    "qwen-3.8-omni-flash-a": {"provider": "openrouter", "model": "qwen/qwen3.8-omni-flash"},
    "jd_kimi_k3": {"provider": "openrouter", "model": "moonshotai/kimi-k3"},
    "kimi-k3": {"provider": "openrouter", "model": "moonshotai/kimi-k3"},
    "jd_gt5_6_sol_b": {"provider": "jdcloud", "model": "gt-5.6-sol-b"},
    "gt-5.6-sol-b": {"provider": "jdcloud", "model": "gt-5.6-sol-b"},
    "jd_gt6_sol_a": {"provider": "jdcloud", "model": "gt-6-sol-a"},
    "gt-6-sol-a": {"provider": "jdcloud", "model": "gt-6-sol-a"},
    "jd_gt6_as_a": {"provider": "jdcloud", "model": "gt-6-as-a"},
    "gt-6-as-a": {"provider": "jdcloud", "model": "gt-6-as-a"},
}


class ApiClientError(RuntimeError):
    """A local configuration, file, or response validation error."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        value = yaml.safe_load(text)
        return value if isinstance(value, dict) else {}
    # A JSON config is valid YAML and is a safe fallback without PyYAML.
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiClientError("PyYAML is required to read non-JSON config.yaml") from exc
    return value if isinstance(value, dict) else {}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load provider/model configuration; an absent file means built-in defaults."""
    return _read_yaml(Path(config_path) if config_path else DEFAULT_CONFIG)


def _model_spec(model: str, config: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Resolve a friendly model name to ``provider``, provider model id and params."""
    models = config.get("models") or {}
    spec: Any = models.get(model) if isinstance(models, Mapping) else None
    if spec is None and isinstance(models, list):
        spec = next((item for item in models if isinstance(item, Mapping) and item.get("name") == model), None)
    if spec is None:
        spec = DEFAULT_MODELS.get(model)
    if isinstance(spec, str):
        spec = {"provider": "jdcloud", "model": spec}
    if not isinstance(spec, Mapping):
        # Passing a provider model id directly is useful for OpenAI-compatible APIs.
        spec = {"provider": "openai", "model": model}
    provider = str(spec.get("provider") or "openai")
    model_id = str(spec.get("model") or spec.get("model_id") or model)
    params = dict(spec.get("params") or {})
    for key in ("max_tokens", "temperature", "top_p", "response_format", "reasoning_effort", "thinking_budget"):
        if key in spec and key not in params:
            params[key] = spec[key]
    return provider, model_id, {**dict(spec), "params": params}


def _provider_spec(provider: str, config: Mapping[str, Any], model_spec: Mapping[str, Any]) -> dict[str, Any]:
    providers = config.get("providers") or {}
    value = providers.get(provider, {}) if isinstance(providers, Mapping) else {}
    result = dict(value) if isinstance(value, Mapping) else {}
    # A model entry can override endpoint/key file for simple one-off setups.
    for key in ("base_url", "api_key_file", "api_key_env", "timeout", "max_attempts", "headers"):
        if key in model_spec:
            result[key] = model_spec[key]
    if provider in {"jd", "jdcloud", "joybuilder"}:
        result.setdefault("base_url", "https://modelservice.jdcloud.com/v1/chat/completions")
        result.setdefault("api_key_file", str(DEFAULT_KEY_FILE))
    else:
        result.setdefault("base_url", "https://api.openai.com/v1/chat/completions")
    result.setdefault("api_key_env", "OPENAI_API_KEY" if provider == "openai" else "API_KEY")
    result.setdefault("timeout", 120)
    result.setdefault("max_attempts", 3)
    return result


def _api_key(spec: Mapping[str, Any]) -> str:
    env_name = str(spec.get("api_key_env") or "")
    value = os.environ.get(env_name, "").strip() if env_name else ""
    key_file = spec.get("api_key_file")
    if not value and key_file:
        path = Path(str(key_file)).expanduser()
        if path.is_file():
            # Accept a bare key or KEY=VALUE; only the value is retained.
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    name, _, raw = line.partition("=")
                    if name.strip().lower() not in {"api_key", "key", env_name.lower()}:
                        continue
                    line = raw.strip()
                value = line.strip().strip("'\"")
                if value:
                    break
    if not value:
        raise ApiClientError("未找到 API Key；请在 config.yaml 指定 api_key_file 或 api_key_env")
    return value


def _mime(path: Path) -> str:
    value = mimetypes.guess_type(path.name)[0]
    if value not in {"image/png", "image/jpeg", "image/jpg"}:
        raise ApiClientError(f"仅支持 PNG/JPEG 图像，实际类型为 {value or 'unknown'}: {path}")
    return "image/jpeg" if value == "image/jpg" else value


def build_payload(model_id: str, prompt: str, image_bytes: bytes, mime: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a Chat Completions multimodal request with a Base64 Data URI."""
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    request_params = dict(params or {})
    image_detail = request_params.pop("image_detail", None)
    image_part = {"type": "image_url", "image_url": {"url": data_uri}}
    if image_detail:
        image_part["image_url"]["detail"] = str(image_detail)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            image_part,
        ]}],
    }
    if request_params:
        payload.update(request_params)
    return payload


def _redact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return request metadata without persisting a huge inline image."""
    result = json.loads(json.dumps(payload, ensure_ascii=False))
    for message in result.get("messages", []):
        for part in message.get("content", []):
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                encoded = url.split(",", 1)[1] if "," in url else ""
                try:
                    raw = base64.b64decode(encoded)
                    part["image_url"]["url"] = f"<inline image {len(raw)} bytes sha256={hashlib.sha256(raw).hexdigest()}>"
                except Exception:
                    part["image_url"]["url"] = "<invalid image data>"
    return result


def _extract_text(body: Any) -> str:
    try:
        content = body["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return str(content or "")


def _usage(body: Any) -> dict[str, Any]:
    usage = body.get("usage", {}) if isinstance(body, Mapping) else {}
    return dict(usage) if isinstance(usage, Mapping) else {}


def _number(value: Any) -> int | None:
    """Return a non-negative integer token count when the provider supplies one."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _usage_summary(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize common usage field variants without discarding raw usage."""
    raw = usage if isinstance(usage, Mapping) else {}
    prompt_tokens = _number(raw.get("prompt_tokens", raw.get("input_tokens")))
    completion_tokens = _number(raw.get("completion_tokens", raw.get("output_tokens")))
    total_tokens = _number(raw.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    prompt_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": _number(prompt_details.get("cached_tokens")) if isinstance(prompt_details, Mapping) else None,
        "reasoning_tokens": _number(completion_details.get("reasoning_tokens")) if isinstance(completion_details, Mapping) else None,
    }


def _pricing(model_spec: Mapping[str, Any], config: Mapping[str, Any], model: str, model_id: str) -> Mapping[str, Any] | None:
    """Find optional USD-per-million-token pricing without inventing provider rates."""
    value = model_spec.get("pricing") if isinstance(model_spec, Mapping) else None
    if value is None:
        prices = config.get("pricing") or {}
        if isinstance(prices, Mapping):
            value = prices.get(model) or prices.get(model_id)
    return value if isinstance(value, Mapping) else None


def _cost_record(usage_summary: Mapping[str, Any], pricing: Mapping[str, Any] | None) -> dict[str, Any]:
    """Record a transparent estimate; unknown pricing remains null."""
    result: dict[str, Any] = {"currency": "USD", "estimated_usd": None, "source": "unconfigured"}
    if not pricing:
        return result
    input_rate = pricing.get("input_usd_per_1m", pricing.get("prompt_usd_per_1m"))
    output_rate = pricing.get("output_usd_per_1m", pricing.get("completion_usd_per_1m"))
    try:
        input_rate = float(input_rate) if input_rate is not None else None
        output_rate = float(output_rate) if output_rate is not None else None
    except (TypeError, ValueError):
        return result
    if (input_rate is None or input_rate < 0) and (output_rate is None or output_rate < 0):
        return result
    prompt = usage_summary.get("prompt_tokens") or 0
    completion = usage_summary.get("completion_tokens") or 0
    estimate = (prompt * (input_rate or 0.0) + completion * (output_rate or 0.0)) / 1_000_000
    result.update({"estimated_usd": round(estimate, 10), "source": "config.pricing",
                   "input_usd_per_1m": input_rate, "output_usd_per_1m": output_rate})
    return result


def _provider_created_at(body: Any) -> str | None:
    value = body.get("created") if isinstance(body, Mapping) else None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return None


def _embedded_error(body: Any) -> Any:
    """Provider errors may be returned inside HTTP 200 choices."""
    if not isinstance(body, Mapping):
        return None
    if body.get("error") and not body.get("choices"):
        return body.get("error")
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(choice, Mapping) and (choice.get("error") or choice.get("finish_reason") == "error"):
        return choice.get("error") or {"message": "finish_reason=error"}
    return None


def _cache_key(image: bytes, model_id: str, prompt: str, params: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(image)
    digest.update(model_id.encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    digest.update(json.dumps(dict(params), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _retry_after(response: Any, attempt: int) -> float:
    try:
        value = float(response.headers.get("Retry-After", ""))
        if value >= 0:
            return min(value, 30.0)
    except (TypeError, ValueError, AttributeError):
        pass
    return min(2.0 * (2 ** (attempt - 1)), 30.0)


def analyze_image(
    image_path: str | Path,
    prompt: str,
    model: str = "jd_gt6",
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
    execute: bool = True,
    cache_dir: str | Path | None = None,
    max_requests: int | None = None,
    mock_response: Mapping[str, Any] | str | Path | None = None,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Analyze one image and return a serialisable call record.

    ``dry_run`` and ``execute=False`` never read a key or make network calls.
    ``mock_response`` exercises downstream parsing without network access.
    """
    request_started_at = datetime.now(timezone.utc).isoformat()
    path = Path(image_path)
    if not path.is_file():
        raise ApiClientError(f"图像不存在: {path}")
    image = path.read_bytes()
    mime = _mime(path)
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG
    config = load_config(config_file)
    provider, model_id, model_spec = _model_spec(model, config)
    provider_spec = _provider_spec(provider, config, model_spec)
    # Relative key paths in config.yaml are resolved next to that config file.
    if provider_spec.get("api_key_file"):
        key_path = Path(str(provider_spec["api_key_file"]))
        if not key_path.is_absolute():
            provider_spec["api_key_file"] = str(config_file.parent / key_path)
    params = dict(model_spec.get("params") or {})
    # Include the endpoint so changing provider routing cannot reuse a response
    # produced by a different service under the same model id.
    key = _cache_key(image, f"{provider}:{model_id}:{provider_spec.get('base_url', '')}", prompt, params)
    pricing = _pricing(model_spec, config, model, model_id)
    redacted: dict[str, Any] | None = None

    if cache_dir and execute and not dry_run:
        cache_path = Path(cache_dir) / f"{key}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("ok"):
                    cached["cached"] = True
                    cached["cache_hit_at_utc"] = datetime.now(timezone.utc).isoformat()
                    cached["cache_path"] = str(cache_path)
                    cached.setdefault("provider_created_at_utc", _provider_created_at(cached.get("response") or cached.get("raw_response")))
                    cached.setdefault("usage_summary", _usage_summary(cached.get("usage")))
                    cached.setdefault("cost", _cost_record(cached["usage_summary"], pricing))
                    return cached
            except (OSError, ValueError):
                pass
    payload = build_payload(model_id, prompt, image, mime, params)
    redacted = _redact_payload(payload)
    common: dict[str, Any] = {
        "ok": False, "status": None, "provider": provider, "model_requested": model,
        "model": model_id, "actual_model": None, "cached": False, "cache_key": key,
        "image": {"path": str(path), "bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()},
        "request": redacted, "usage": {}, "usage_summary": _usage_summary({}),
        "cost": _cost_record(_usage_summary({}), pricing), "attempts": [], "request_count": 0, "latency_s": 0.0,
        "request_started_at_utc": request_started_at, "request_finished_at_utc": None,
    }
    if dry_run or not execute:
        common.update({"status": "dry_run", "dry_run": True, "message": "未发送 API 请求", "request_finished_at_utc": datetime.now(timezone.utc).isoformat()})
        return common

    if mock_response is not None:
        if isinstance(mock_response, (str, Path)):
            body = json.loads(Path(mock_response).read_text(encoding="utf-8"))
        else:
            body = dict(mock_response)
        embedded = _embedded_error(body)
        mock_ok = isinstance(body, Mapping) and bool(body.get("choices")) and not embedded
        common.update({"ok": mock_ok, "status": 200, "response": body, "raw_response": body,
                       "text": _extract_text(body), "usage": _usage(body),
                       "actual_model": body.get("model") if isinstance(body, Mapping) else None,
                       "provider_created_at_utc": _provider_created_at(body),
                       "mock": True, "request_count": 0, "request_finished_at_utc": datetime.now(timezone.utc).isoformat()})
        common["usage_summary"] = _usage_summary(common["usage"])
        common["cost"] = _cost_record(common["usage_summary"], pricing)
        if not mock_ok:
            common["error"] = embedded or "模型响应无 choices"
            common["error_type"] = "api_error" if embedded else "response_schema_error"
        return common

    endpoint = str(provider_spec["base_url"])
    timeout = float(provider_spec.get("timeout", 120))
    try:
        configured_attempts = int(provider_spec.get("max_attempts", 3))
    except (TypeError, ValueError) as exc:
        raise ApiClientError("provider.max_attempts 必须是整数") from exc
    configured_attempts = max(1, configured_attempts)
    if max_requests is None:
        request_limit = configured_attempts
    else:
        try:
            request_limit = int(max_requests)
        except (TypeError, ValueError) as exc:
            raise ApiClientError("max_requests 必须是整数") from exc
        if request_limit < 1:
            raise ApiClientError("max_requests 必须 >= 1")
        request_limit = min(request_limit, configured_attempts)
    attempts_limit = max(1, request_limit)
    # Validate the local request budget before reading credentials or touching
    # the network.  A rejected budget must be side-effect free.
    key_value = _api_key(provider_spec)
    headers = {"Authorization": f"Bearer {key_value}", "Content-Type": "application/json"}
    headers.update(dict(provider_spec.get("headers") or {}))
    started = time.monotonic()
    sender = post or requests.post
    body: Any = None
    body_text: str | None = None
    status: int | None = None
    error: Any = None
    for attempt in range(1, attempts_limit + 1):
        t0 = time.monotonic()
        response = None
        try:
            response = sender(endpoint, headers=headers, json=payload, timeout=timeout)
            status = int(response.status_code)
            try:
                body = response.json()
                body_text = None
            except (ValueError, AttributeError):
                body = None
                body_text = str(getattr(response, "text", ""))[:4000]
            embedded = _embedded_error(body)
            error = embedded or (None if status < 400 else body_text or f"HTTP {status}")
            good = status == 200 and isinstance(body, Mapping) and not embedded and bool(body.get("choices"))
            common["attempts"].append({"attempt": attempt, "status": status, "latency_s": time.monotonic() - t0, "error": error})
            if good:
                common.update({"ok": True, "response": body, "raw_response": body, "text": _extract_text(body), "usage": _usage(body), "actual_model": body.get("model"), "provider_created_at_utc": _provider_created_at(body)})
                common["usage_summary"] = _usage_summary(common["usage"])
                common["cost"] = _cost_record(common["usage_summary"], pricing)
                break
        except requests.RequestException as exc:
            status = None
            error = f"{type(exc).__name__}: {exc}"
            common["attempts"].append({"attempt": attempt, "status": None, "latency_s": time.monotonic() - t0, "error": error})
        retryable = status in RETRY_STATUS or status is None or (status == 200 and not common.get("ok"))
        if not retryable or attempt >= attempts_limit:
            break
        time.sleep(_retry_after(response, attempt))
    common["latency_s"] = time.monotonic() - started
    common["request_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    common["request_count"] = len(common["attempts"])
    common["status"] = status
    if not common.get("ok"):
        common["error"] = error or "模型响应无 choices"
        if _embedded_error(body):
            common["error_type"] = "api_error"
        elif body is None and body_text is not None:
            common["error_type"] = "response_parse_error"
        elif status is None or (status is not None and status >= 400):
            common["error_type"] = "api_error"
        else:
            common["error_type"] = "response_schema_error"
        if body is not None:
            common["response"] = body
            common["raw_response"] = body
        if body_text is not None:
            common["response_text"] = body_text
            common["raw_response"] = body_text
    if common.get("ok") and cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(common, ensure_ascii=False, indent=2), encoding="utf-8")
    return common


__all__ = ["ApiClientError", "HARD_MAX_REQUESTS", "analyze_image", "build_payload", "load_config"]
