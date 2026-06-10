"""Capture and format vLLM Semantic Router decision headers.

When Hermes talks to the Semantic Router's OpenAI-compatible proxy (the
``router`` provider, ``/model router``), the proxy returns its routing decision
as ``x-vsr-*`` response headers on each ``/v1/chat/completions`` call, e.g.::

    x-vsr-selected-model: Qwen3.5-9B-NoThinking
    x-vsr-selected-decision: route_security_guard
    x-vsr-selected-confidence: 1.0000
    x-vsr-matched-structure: short_query,any_query
    x-vsr-matched-complexity: needs_reasoning:medium
    x-vsr-context-token-count: 3
    x-vsr-looper-models-used: Qwen3.5-9B-NoThinking
    x-vsr-looper-iterations: 1
    x-vsr-looper-algorithm: confidence

The OpenAI SDK discards response headers on the success path, and Hermes streams
by default, so we capture them with an httpx ``response`` event hook installed on
the OpenAI client's underlying httpx client (``attach_capture_hook``, called from
``agent_runtime_helpers.create_openai_client``).

Storage is **on the agent** (``agent._last_vsr_headers``), not a thread-local:
Hermes runs the actual API request on a daemon worker thread
(``_interruptible_api_call``), so the hook fires on a different thread than the
conversation loop that reads it. The agent object is shared across both threads.

The hook must never raise — an exception in an httpx event hook would break the
request — so everything here is defensively wrapped.
"""

import functools
from typing import Any, Dict, Optional

_PREFIX = "x-vsr-"


def _extract(response: Any) -> Optional[Dict[str, str]]:
    """Pull ``x-vsr-*`` headers from an httpx response into a dict, or None."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    vsr = {
        k[len(_PREFIX):]: v
        for k, v in headers.items()
        if k.lower().startswith(_PREFIX)
    }
    return vsr or None


def capture_for_agent(agent: Any, response: Any) -> None:
    """httpx ``response`` event hook (bound to ``agent`` via functools.partial).

    Stashes any ``x-vsr-*`` headers on ``agent._last_vsr_headers`` (None when the
    response isn't from the router). Safe for streaming responses — only reads
    ``.headers``, never the body — and swallows all errors.
    """
    try:
        agent._last_vsr_headers = _extract(response)
    except Exception:
        pass


def attach_capture_hook(openai_client: Any, agent: Any) -> None:
    """Install the per-agent capture hook on an OpenAI client's httpx transport.

    Idempotent per httpx client (marks it with ``_vsr_hooked``). Covers both the
    keepalive client Hermes injects and any client the SDK builds itself. Never
    raises.
    """
    try:
        httpx_client = getattr(openai_client, "_client", None)
        if httpx_client is None or getattr(httpx_client, "_vsr_hooked", False):
            return
        hooks = httpx_client.event_hooks
        hooks.setdefault("response", []).append(
            functools.partial(capture_for_agent, agent)
        )
        httpx_client._vsr_hooked = True
    except Exception:
        pass


def take_last(agent: Any) -> Optional[Dict[str, str]]:
    """Return and clear the most recent call's ``x-vsr-*`` headers for ``agent``."""
    vsr = getattr(agent, "_last_vsr_headers", None)
    try:
        agent._last_vsr_headers = None
    except Exception:
        pass
    return vsr or None


def record_router_usage(agent: Any, vsr: Dict[str, str]) -> None:
    """Use the router's real token count from ``x-vsr-*`` headers.

    The proxy returns ``usage: {0,0,0}``, so Hermes' context bar would otherwise
    fall back to a tool-schema-inflated estimate. ``x-vsr-context-token-count``
    is the real input-token count the backend processed (matches lemonade's
    telemetry), so we (1) feed it into the context compressor to make the live
    bar accurate, and (2) accumulate per-routed-model totals on the agent for
    the status-bar breakdown. Never raises.
    """
    try:
        try:
            ctx = int(vsr.get("context-token-count"))
        except (TypeError, ValueError):
            return
        model = vsr.get("selected-model") or vsr.get("looper-model") or "?"

        # (1) Make the live context bar reflect real tokens, not the estimate.
        comp = getattr(agent, "context_compressor", None)
        if comp is not None and ctx > 0:
            comp.last_prompt_tokens = ctx
            comp.last_real_prompt_tokens = ctx

        # (2) Per-model cumulative tracker (input tokens + call count).
        usage = getattr(agent, "_router_model_usage", None)
        if usage is None:
            usage = {}
            agent._router_model_usage = usage
        entry = usage.setdefault(model, {"input_tokens": 0, "calls": 0})
        entry["input_tokens"] += ctx
        entry["calls"] += 1
    except Exception:
        pass


def format_summary(vsr: Dict[str, str]) -> str:
    """Build a compact one-line summary of a router decision for display."""
    model = vsr.get("selected-model") or vsr.get("looper-model") or "?"
    parts = [f"🧭 router → {model}"]

    decision = vsr.get("selected-decision")
    conf = vsr.get("selected-confidence")
    if decision:
        seg = f"decision={decision}"
        if conf:
            try:
                seg += f" (conf {float(conf):.2f})"
            except (TypeError, ValueError):
                seg += f" (conf {conf})"
        parts.append(seg)

    signals = []
    if vsr.get("matched-structure"):
        signals.append(f"structure={vsr['matched-structure']}")
    if vsr.get("matched-complexity"):
        signals.append(f"complexity={vsr['matched-complexity']}")
    if vsr.get("matched-jailbreak"):
        signals.append(f"jailbreak={vsr['matched-jailbreak']}")
    if vsr.get("matched-pii"):
        signals.append(f"pii={vsr['matched-pii']}")
    if vsr.get("context-token-count"):
        signals.append(f"ctx_tokens={vsr['context-token-count']}")
    if signals:
        parts.append(" ".join(signals))

    loop = []
    if vsr.get("looper-algorithm"):
        loop.append(f"algo={vsr['looper-algorithm']}")
    if vsr.get("looper-iterations"):
        loop.append(f"iters={vsr['looper-iterations']}")
    models_used = vsr.get("looper-models-used")
    if models_used and models_used != model:
        loop.append(f"models={models_used}")
    if loop:
        parts.append(" ".join(loop))

    return " | ".join(parts)
