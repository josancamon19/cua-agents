"""OpenAI CUA agent — uses GPT-5.x with the computer tool via the Responses API.

Implements the same predict/reset interface as the other agents, converting OpenAI's
structured computer actions into pyautogui code for VM execution.
"""

import base64
import logging
import time
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

OPERATOR_PROMPT_TEMPLATE = (Path(__file__).with_name("prompt.txt")).read_text().strip()

# Key mapping from OpenAI key names to pyautogui key names
_KEY_MAP = {
    "enter": "enter",
    "return": "enter",
    "escape": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "tab": "tab",
    "space": "space",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "home": "home",
    "end": "end",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "meta": "win",
    "super": "win",
    "command": "command",
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "f6": "f6",
    "f7": "f7",
    "f8": "f8",
    "f9": "f9",
    "f10": "f10",
    "f11": "f11",
    "f12": "f12",
}

API_RETRY_TIMES = 12
API_RETRY_BASE = 2  # exponential backoff: 2, 4, 8, 16... capped at 60s


def _get(action, key, default=None):
    """Get attribute from an action object or dict."""
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


def _usage_int(obj, key: str) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        value = obj.get(key, 0)
    else:
        value = getattr(obj, key, 0)
    return int(value or 0)


def _usage_detail_int(usage, details_key: str, key: str) -> int:
    details = None
    if isinstance(usage, dict):
        details = usage.get(details_key)
    else:
        details = getattr(usage, details_key, None)
    return _usage_int(details, key)


def _action_to_pyautogui(action) -> str:
    """Convert an OpenAI CUA action to pyautogui code."""
    action_type = _get(action, "type")

    if action_type == "click":
        x, y = _get(action, "x"), _get(action, "y")
        button = _get(action, "button", "left")
        # SDK Click.button: "left", "right", "wheel", "back", "forward"
        fn_map = {"left": "click", "right": "rightClick", "wheel": "middleClick"}
        fn = fn_map.get(button, "click")
        return f"pyautogui.{fn}({x}, {y})\n"

    if action_type == "double_click":
        x, y = _get(action, "x"), _get(action, "y")
        return f"pyautogui.doubleClick({x}, {y})\n"

    if action_type == "scroll":
        x, y = _get(action, "x"), _get(action, "y")
        scroll_x = _get(action, "scroll_x", 0)
        scroll_y = _get(action, "scroll_y", 0)
        code = ""
        if scroll_y != 0:
            code += f"pyautogui.scroll({scroll_y}, {x}, {y})\n"
        if scroll_x != 0:
            code += f"pyautogui.hscroll({scroll_x}, {x}, {y})\n"
        return code or f"pyautogui.scroll(0, {x}, {y})\n"

    if action_type == "keypress":
        keys = _get(action, "keys", [])
        mapped = [_KEY_MAP.get(k.lower(), k.lower()) for k in keys]
        if len(mapped) == 1:
            return f"pyautogui.press('{mapped[0]}')\n"
        return f"pyautogui.hotkey({', '.join(repr(k) for k in mapped)})\n"

    if action_type == "type":
        text = _get(action, "text", "")
        # Split into segments: regular text (use typewrite) vs special chars
        lines = text.split("\n")
        result = ""
        for i, line in enumerate(lines):
            if line:
                # typewrite only handles ASCII; fall back to press for non-ASCII
                if line.isascii() and "\t" not in line:
                    result += f"pyautogui.typewrite({line!r}, interval=0.02)\n"
                else:
                    for char in line:
                        if char == "\t":
                            result += "pyautogui.press('tab')\n"
                        else:
                            result += f"pyautogui.press({char!r})\n"
            if i < len(lines) - 1:
                result += "pyautogui.press('enter')\n"
        return result or "time.sleep(0.1)\n"

    if action_type == "drag":
        path = _get(action, "path", [])
        if len(path) < 2:
            return "time.sleep(0.1)\n"
        start = path[0]
        end = path[-1]
        sx, sy = _get(start, "x"), _get(start, "y")
        ex, ey = _get(end, "x"), _get(end, "y")
        return f"pyautogui.moveTo({sx}, {sy})\npyautogui.dragTo({ex}, {ey}, duration=0.5)\n"

    if action_type == "move":
        return f"pyautogui.moveTo({_get(action, 'x')}, {_get(action, 'y')})\n"

    if action_type == "wait":
        return "time.sleep(2)\n"

    if action_type == "screenshot":
        return "time.sleep(0.1)\n"

    logger.warning("Unknown OpenAI CUA action type: %s", action_type)
    return "time.sleep(0.1)\n"


class OpenAICUAAgent:
    """OpenAI computer-use agent using the Responses API with {"type": "computer"}."""

    def __init__(
        self,
        model: str = "gpt-5.4-2026-03-05",
        screen_size: tuple[int, int] = (1920, 1080),
        password: str = "",
        **_kwargs,
    ):
        self.model = model
        self.screen_size = screen_size
        self.password = password
        self._client = OpenAI()
        self._response_id: str | None = None
        self._pending_call_id: str | None = None
        self._pending_safety_checks: list[dict] = []
        self._step = 0
        self._last_usage: dict = {}

    def reset(self) -> None:
        self._response_id = None
        self._pending_call_id = None
        self._pending_safety_checks = []
        self._step = 0

    def predict(self, instruction: str, obs: dict, **kwargs) -> tuple[str, list[str]]:
        """Return (raw_response, [pyautogui_code_or_signal])."""
        screenshot_bytes = obs["screenshot"]
        b64 = base64.b64encode(screenshot_bytes).decode()
        media_type = obs.get("media_type", "image/jpeg")
        image_url = f"data:{media_type};base64,{b64}"

        tools = [{"type": "computer"}]

        response = None
        reasoning = {"effort": "high", "summary": "detailed"}
        for attempt in range(API_RETRY_TIMES):
            try:
                if self._step == 0:
                    operator_prompt = OPERATOR_PROMPT_TEMPLATE.replace("{PASSWORD}", self.password)
                    response = self._client.responses.create(
                        model=self.model,
                        tools=tools,
                        reasoning=reasoning,
                        truncation="auto",
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_image", "image_url": image_url},
                                    {"type": "input_text", "text": instruction + "\n" + operator_prompt},
                                ],
                            }
                        ],
                    )
                else:
                    call_output = {
                        "type": "computer_call_output",
                        "call_id": self._pending_call_id,
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": image_url,
                        },
                    }
                    # Acknowledge any pending safety checks from the previous response
                    if self._pending_safety_checks:
                        call_output["acknowledged_safety_checks"] = [
                            {"id": sc["id"], "code": sc.get("code", ""), "message": sc.get("message", "")}
                            for sc in self._pending_safety_checks
                        ]
                        self._pending_safety_checks = []

                    response = self._client.responses.create(
                        model=self.model,
                        previous_response_id=self._response_id,
                        tools=tools,
                        reasoning=reasoning,
                        truncation="auto",
                        input=[call_output],
                    )
                break
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                delay = min(API_RETRY_BASE * (2**attempt), 60)
                logger.warning(
                    "OpenAI transient error (attempt %d/%d, retry in %ds): %s", attempt + 1, API_RETRY_TIMES, delay, e
                )
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(delay)
                else:
                    logger.error("All OpenAI API attempts failed")
                    return str(e), ["FAIL"]
            except Exception as e:
                logger.error("OpenAI non-retryable error: %s", e)
                return str(e), ["FAIL"]

        if response is None:
            return "No response", ["FAIL"]

        self._response_id = response.id
        self._step += 1

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = _usage_int(usage, "input_tokens")
            cached_tokens = _usage_detail_int(usage, "input_tokens_details", "cached_tokens")
            output_tokens = _usage_int(usage, "output_tokens")
            reasoning_tokens = _usage_detail_int(usage, "output_tokens_details", "reasoning_tokens")
            self._last_usage = {
                # OpenAI reports cached input inside input_tokens. Normalize to the
                # same shape as Anthropic: fresh input is separated from cache read.
                "input_tokens": max(input_tokens - cached_tokens, 0),
                "cache_read_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "total_input_tokens": input_tokens,
            }
            if reasoning_tokens:
                self._last_usage["reasoning_output_tokens"] = reasoning_tokens
        else:
            self._last_usage = {}

        # Parse response output items
        raw_parts: list[str] = []
        actions: list[str] = []
        found_computer_call = False

        for item in response.output:
            if item.type == "computer_call":
                found_computer_call = True
                self._pending_call_id = item.call_id
                # Store pending safety checks to acknowledge in the next call
                self._pending_safety_checks = [
                    {"id": sc.id, "code": sc.code, "message": sc.message} for sc in (item.pending_safety_checks or [])
                ]
                if self._pending_safety_checks:
                    logger.info("Safety checks to acknowledge: %s", self._pending_safety_checks)
                # GA computer tool returns batched actions[] array
                call_actions = item.actions or []
                # Fallback: single action field (older API responses)
                if not call_actions and getattr(item, "action", None):
                    call_actions = [item.action]
                for act in call_actions:
                    code = _action_to_pyautogui(act)
                    actions.append(code)
                    raw_parts.append(f"[COMPUTER_CALL] {act}")
            elif item.type == "text":
                raw_parts.append(f"[TEXT] {item.text}")
            elif item.type == "reasoning":
                for summary_item in getattr(item, "summary", []):
                    raw_parts.append(f"[REASONING] {getattr(summary_item, 'text', '')}")

        raw_str = "\n".join(raw_parts)

        if not found_computer_call:
            if self._step == 1 and any("error" in p.lower() for p in raw_parts):
                logger.error("First response was an error: %s", raw_str[:500])
                actions = ["FAIL"]
            else:
                actions = ["DONE"]

        return raw_str, actions
