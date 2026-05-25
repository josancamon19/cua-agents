"""Anthropic CUA agent — uses Claude's native computer-use tool.

Ported from OSWorld's mm_agents/anthropic/ with simplifications for our environment.
"""

import base64
import io
import logging
import os
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic, APIError, APIResponseValidationError, APIStatusError
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)
from PIL import Image

logger = logging.getLogger(__name__)

COMPUTER_USE_BETA_FLAG = "computer-use-2025-11-24"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

# Internal resolution used by Anthropic's computer tool
INTERNAL_WIDTH = 1280
INTERNAL_HEIGHT = 720

API_RETRY_TIMES = 100
API_RETRY_INTERVAL = 5

SYSTEM_PROMPT_TEMPLATE = (Path(__file__).with_name("prompt.txt")).read_text().strip()

# Key mapping for converting Anthropic tool actions to pyautogui
_KEY_MAP = {
    "page_down": "pagedown",
    "page_up": "pageup",
    "super_l": "win",
    "super": "command",
    "escape": "esc",
}


# ── Utilities (ported from OSWorld mm_agents/anthropic/utils.py) ─────────────


def _response_to_params(response) -> list[BetaContentBlockParam]:
    """Convert API response to message params for history."""
    res: list[BetaContentBlockParam] = []
    if not response.content:
        return res
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                thinking_block: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(messages: list[BetaMessageParam]) -> None:
    """Set cache breakpoints for the 2 most recent user turns."""
    # Clear all existing cache_control to prevent accumulation across calls
    for message in messages:
        if isinstance(content := message.get("content"), list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)  # type: ignore[union-attr]

    breakpoints_remaining = 2
    for message in reversed(messages):
        if breakpoints_remaining <= 0:
            break
        if message["role"] == "user" and isinstance(content := message["content"], list):
            content[-1]["cache_control"] = BetaCacheControlEphemeralParam({"type": "ephemeral"})  # type: ignore[typeddict-unknown-key]
            breakpoints_remaining -= 1


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
) -> None:
    """Remove old screenshot images from tool results to manage context size."""
    if images_to_keep is None:
        return
    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )
    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )
    images_to_remove = total_images - images_to_keep
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


# ── Agent ────────────────────────────────────────────────────────────────────


class AnthropicCUAAgent:
    """Anthropic computer-use agent using Claude's native computer tool."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        screen_size: tuple[int, int] = (1920, 1080),
        only_n_most_recent_images: int = 10,
        password: str = "",
        no_thinking: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.screen_size = screen_size
        self.only_n_most_recent_images = only_n_most_recent_images
        self.password = password
        self.no_thinking = no_thinking
        self.temperature = temperature
        self.top_p = top_p
        self.messages: list[BetaMessageParam] = []
        self.resize_factor = (screen_size[0] / INTERNAL_WIDTH, screen_size[1] / INTERNAL_HEIGHT)
        self._last_usage: dict = {}

    def reset(self) -> None:
        """Reset agent state."""
        self.messages = []

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _set_observed_screen_size(self, size: tuple[int, int]) -> None:
        """Use the real screenshot dimensions for coordinate scaling back to the VM."""
        width, height = size
        if width <= 0 or height <= 0:
            return
        self.screen_size = (int(width), int(height))
        self.resize_factor = (self.screen_size[0] / INTERNAL_WIDTH, self.screen_size[1] / INTERNAL_HEIGHT)

    def _resize_screenshot(self, screenshot_bytes: bytes) -> bytes:
        """Resize screenshot to internal resolution (1280x720)."""
        img = Image.open(io.BytesIO(screenshot_bytes))
        self._set_observed_screen_size(img.size)
        resized = img.resize((INTERNAL_WIDTH, INTERNAL_HEIGHT), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()

    def _scale_coordinate(self, coordinate: Sequence[int | float]) -> tuple[int, int]:
        """Map Anthropic's 1280x720 tool coordinate back to the observed screen."""
        x = round(float(coordinate[0]) * self.resize_factor[0])
        y = round(float(coordinate[1]) * self.resize_factor[1])
        width, height = self.screen_size
        return (max(0, min(x, width - 1)), max(0, min(y, height - 1)))

    def _add_tool_result(self, tool_call_id: str, result_text: str, screenshot: bytes | None = None) -> None:
        """Add a tool result to message history."""
        content: list[dict] = [{"type": "text", "text": result_text}]
        if screenshot is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(screenshot).decode(),
                    },
                }
            )
        self.messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content}],
            }
        )

    def _tool_call_to_pyautogui(self, tool_call: dict) -> str:
        """Convert a computer tool call to pyautogui code string."""
        args = tool_call["input"]
        action = args.get("action", "")
        text = args.get("text")
        coordinate = args.get("coordinate")
        start_coordinate = args.get("start_coordinate")
        scroll_direction = args.get("scroll_direction")
        scroll_amount = args.get("scroll_amount")
        duration = args.get("duration")

        # Scale coordinates from internal (1280x720) to actual screen size
        if coordinate:
            coordinate = self._scale_coordinate(coordinate)
        if start_coordinate:
            start_coordinate = self._scale_coordinate(start_coordinate)

        # Normalize action names
        action = {"left click": "click", "right click": "right_click", "left_click": "click"}.get(action, action)

        result = ""

        # ── Terminal actions ─────────────────────────────────────────────
        if action == "done":
            return "DONE"
        if action == "fail":
            return "FAIL"
        if action == "wait":
            return "time.sleep(0.5)\n"
        if action == "screenshot":
            return "time.sleep(0.1)\n"

        # ── Mouse move / drag ────────────────────────────────────────────
        if action == "mouse_move" and coordinate:
            x, y = coordinate
            result = f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"

        elif action == "left_click_drag" and coordinate:
            x, y = coordinate
            if start_coordinate:
                sx, sy = start_coordinate
                result = f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})\n"
            result += f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"

        # ── Keyboard ─────────────────────────────────────────────────────
        elif action == "key" and text:
            keys = text.split("+")
            for key in keys:
                k = _KEY_MAP.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyDown('{k}')\n"
            for key in reversed(keys):
                k = _KEY_MAP.get(key.strip().lower(), key.strip().lower())
                result += f"pyautogui.keyUp('{k}')\n"

        elif action == "type" and text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line:
                    if line.isascii():
                        result += f"pyautogui.typewrite({line!r}, interval=0.02)\n"
                    else:
                        for char in line:
                            result += f"pyautogui.press({char!r})\n"
                if i < len(lines) - 1:
                    result += "pyautogui.press('enter')\n"

        elif action == "hold_key" and text:
            for key in text.split("+"):
                result += f"pyautogui.keyDown('{key.strip().lower()}')\n"

        # ── Scroll ───────────────────────────────────────────────────────
        elif action == "scroll":
            if text:
                result += f"pyautogui.keyDown('{text.lower()}')\n"
            scroll_val = scroll_amount if scroll_direction == "up" else -scroll_amount
            if scroll_direction in ("up", "down"):
                if coordinate:
                    result += f"pyautogui.scroll({scroll_val}, {coordinate[0]}, {coordinate[1]})\n"
                else:
                    result += f"pyautogui.scroll({scroll_val})\n"
            else:
                hval = scroll_amount if scroll_direction == "right" else -scroll_amount
                if coordinate:
                    result += f"pyautogui.hscroll({hval}, {coordinate[0]}, {coordinate[1]})\n"
                else:
                    result += f"pyautogui.hscroll({hval})\n"
            if text:
                result += f"pyautogui.keyUp('{text.lower()}')\n"

        # ── Click variants ───────────────────────────────────────────────
        elif action in ("click", "right_click", "double_click", "middle_click", "triple_click"):
            click_fn = {
                "click": "click",
                "right_click": "rightClick",
                "double_click": "doubleClick",
                "middle_click": "middleClick",
                "triple_click": "tripleClick",
            }[action]

            # Hold modifier keys if specified
            if text:
                for key in text.split("+"):
                    result += f"pyautogui.keyDown('{key.strip().lower()}')\n"

            if coordinate:
                result += f"pyautogui.{click_fn}({coordinate[0]}, {coordinate[1]})\n"
            else:
                result += f"pyautogui.{click_fn}()\n"

            if text:
                for key in reversed(text.split("+")):
                    result += f"pyautogui.keyUp('{key.strip().lower()}')\n"

        # ── Mouse button ─────────────────────────────────────────────────
        elif action == "left_mouse_down":
            result = "pyautogui.mouseDown()\n"
        elif action == "left_mouse_up":
            result = "pyautogui.mouseUp()\n"

        else:
            logger.warning("Unknown action: %s (args=%s)", action, args)
            result = "time.sleep(0.1)\n"

        return result

    def _extract_raw_response(self, response) -> str:
        """Extract raw response text for logging."""
        parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                parts.append(f"[TEXT] {block.text}")
            elif hasattr(block, "thinking") and block.thinking:
                parts.append(f"[THINKING] {block.thinking}")
            elif hasattr(block, "name") and hasattr(block, "input"):
                parts.append(f"[TOOL_USE] {block.name}: {block.input}")
        return "\n".join(parts)

    def _get_sampling_params(self) -> dict:
        """Build sampling kwargs for API call."""
        params: dict[str, float] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        return params

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(self, instruction: str, obs: dict, **kwargs) -> tuple[str, list[str]]:
        """Return (raw_response, [pyautogui_code_or_signal, ...])."""
        # Resize screenshot to internal resolution
        screenshot = self._resize_screenshot(obs["screenshot"])

        # Build system prompt
        system = BetaTextBlockParam(
            type="text",
            text=SYSTEM_PROMPT_TEMPLATE.replace("{DATE}", datetime.today().strftime("%A, %B %d, %Y")).replace(
                "{PASSWORD}", self.password
            ),
        )

        # First message: task instruction before screenshot improves target localization.
        if not self.messages:
            b64 = base64.b64encode(screenshot).decode()
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    ],
                }
            )
        else:
            # Add tool results for previous tool_use blocks
            last_content = self.messages[-1]["content"]
            tool_use_blocks = [b for b in last_content if isinstance(b, dict) and b.get("type") == "tool_use"]
            for i, block in enumerate(tool_use_blocks):
                is_last = i == len(tool_use_blocks) - 1
                self._add_tool_result(block["id"], "Success", screenshot=screenshot if is_last else None)

        # Configure client
        api_key = os.environ.get("ANTHROPIC_API_KEY") or None
        auth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or None
        client_kwargs: dict[str, Any] = {"max_retries": 4}
        if api_key:
            client_kwargs["api_key"] = api_key
        elif auth_token:
            client_kwargs["auth_token"] = auth_token
        client = Anthropic(**client_kwargs).with_options(default_headers={"anthropic-beta": COMPUTER_USE_BETA_FLAG})

        # Betas & caching
        betas = [COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG]
        _inject_prompt_caching(self.messages)
        system["cache_control"] = {"type": "ephemeral"}  # type: ignore[typeddict-unknown-key]

        # Image history management
        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                self.messages, self.only_n_most_recent_images, min_removal_threshold=20
            )

        # Tool config
        tools: list[dict[str, Any]] = [
            {
                "name": "computer",
                "type": "computer_20251124",
                "display_width_px": INTERNAL_WIDTH,
                "display_height_px": INTERNAL_HEIGHT,
                "display_number": 1,
            },
        ]

        # Thinking mode
        if self.no_thinking:
            extra_body: dict[str, Any] = {}
            actual_max_tokens = self.max_tokens
        else:
            budget = 2048
            actual_max_tokens = max(self.max_tokens, budget + 500)
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": budget}}

        # API call with retry
        response = None
        for attempt in range(API_RETRY_TIMES):
            try:
                response = client.beta.messages.create(
                    max_tokens=actual_max_tokens,
                    messages=self.messages,
                    model=self.model,
                    system=[system],
                    tools=tools,
                    betas=betas,
                    extra_body=extra_body,
                    **self._get_sampling_params(),
                )
                break
            except (APIError, APIStatusError, APIResponseValidationError) as e:
                logger.warning("API error (attempt %d/%d): %s", attempt + 1, API_RETRY_TIMES, e)
                if isinstance(e, APIStatusError) and e.status_code in {401, 403}:
                    return str(e), ["FAIL"]
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(API_RETRY_INTERVAL)
                else:
                    logger.error("All API attempts failed")
                    return str(e), ["FAIL"]

        if response is None:
            return "No response", ["FAIL"]

        usage = response.usage
        self._last_usage = {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        }

        # Parse response
        response_params = _response_to_params(response)
        raw_str = self._extract_raw_response(response)

        # Store response in message history
        self.messages.append({"role": "assistant", "content": response_params})

        # Check for infeasible
        if "[INFEASIBLE]" in raw_str:
            logger.info("Detected [INFEASIBLE] pattern, signaling FAIL")
            return raw_str, ["FAIL"]

        # Extract actions from tool_use blocks
        actions: list[str] = []
        for block in response_params:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                code = self._tool_call_to_pyautogui(block)
                actions.append(code)

        if not actions:
            actions = ["DONE"]

        return raw_str, actions
