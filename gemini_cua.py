"""Gemini CUA agent — uses Google's computer-use tool via the google-genai SDK.

Converts Gemini's normalized (0-999) coordinate actions into pyautogui code
for VM execution, matching the predict/reset interface of other agents.
"""

import io
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

API_RETRY_TIMES = 5
API_RETRY_INTERVAL = 5
MAX_RECENT_TURN_WITH_SCREENSHOTS = 3

# Actions that don't make sense for a desktop VM (we can't control the browser directly)
_EXCLUDED_ACTIONS = ["navigate", "go_back", "go_forward", "open_web_browser", "search"]
_PREDEFINED_COMPUTER_USE_FUNCTIONS = {
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
}


def _usage_int(usage_meta, name: str) -> int:
    value = getattr(usage_meta, name, 0) if usage_meta else 0
    return int(value or 0)


SYSTEM_PROMPT_TEMPLATE = (Path(__file__).with_name("prompt.txt")).read_text().strip()
GEMINI_DESKTOP_NOTE = """
Important for Gemini Computer Use: even though the API environment is named browser,
this harness executes your clicks and keyboard actions on a full Windows desktop.
You can use Chrome downloads, File Explorer, ZIP folders, PDF/XML viewers, and
Windows shortcuts such as Ctrl+J for downloads and Windows+E for File Explorer.
Downloaded ZIPs are automatically extracted into same-named folders under
Downloads; prefer those extracted folders when reading PDF or XML files.
When opening a file or folder, use double_click_at instead of two click_at calls.
Do not declare infeasible only because an invoice is inside a downloaded ZIP.
Open or extract the ZIP through the Windows UI and continue the accounting task.
The harness automatically approves required safety confirmations inside this
isolated evaluation sandbox.
""".strip()


def _denormalize(coord: int, screen_dim: int) -> int:
    """Convert Gemini's 0-999 normalized coordinate to pixel coordinate."""
    return int(coord / 1000 * screen_dim)


def _action_to_pyautogui(name: str, args: dict, screen_w: int, screen_h: int) -> str:
    """Convert a Gemini CUA function call into pyautogui code."""
    x = _denormalize(args.get("x", 0), screen_w)
    y = _denormalize(args.get("y", 0), screen_h)

    if name == "click_at":
        return f"pyautogui.click({x}, {y})\n"

    if name == "double_click_at":
        return f"pyautogui.doubleClick({x}, {y})\n"

    if name == "right_click_at":
        return f"pyautogui.rightClick({x}, {y})\n"

    if name == "type_text_at":
        text = args.get("text", "")
        clear = args.get("clear_before_typing", True)
        enter = args.get("press_enter", False)
        result = f"pyautogui.click({x}, {y})\n"
        if clear:
            result += "pyautogui.hotkey('ctrl', 'a')\npyautogui.press('backspace')\n"
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line:
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
        if enter:
            result += "pyautogui.press('enter')\n"
        return result

    if name == "hover_at":
        return f"pyautogui.moveTo({x}, {y})\n"

    if name == "scroll_document":
        direction = args.get("direction", "down")
        amount = 5
        if direction == "up":
            return f"pyautogui.scroll({amount})\n"
        elif direction == "down":
            return f"pyautogui.scroll(-{amount})\n"
        elif direction == "left":
            return f"pyautogui.hscroll(-{amount})\n"
        elif direction == "right":
            return f"pyautogui.hscroll({amount})\n"

    if name == "scroll_at":
        direction = args.get("direction", "down")
        # Gemini magnitude is 0-999 (default 800); map to pyautogui scroll clicks
        magnitude_raw = args.get("magnitude", 800)
        clicks = max(1, int(magnitude_raw / 100))  # ~800 → 8 clicks, ~200 → 2 clicks
        if direction in ("up", "down"):
            val = clicks if direction == "up" else -clicks
            return f"pyautogui.scroll({val}, {x}, {y})\n"
        hval = clicks if direction == "right" else -clicks
        return f"pyautogui.hscroll({hval}, {x}, {y})\n"

    if name == "key_combination":
        keys_str = args.get("keys", "")
        keys = [k.strip().lower() for k in keys_str.split("+")]
        key_map = {
            "control": "ctrl",
            "meta": "win",
            "command": "win",
            "windows": "win",
            "arrowup": "up",
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
            "escape": "esc",
        }
        mapped = [key_map.get(k, k) for k in keys]
        if len(mapped) == 1:
            return f"pyautogui.press('{mapped[0]}')\n"
        return f"pyautogui.hotkey({', '.join(repr(k) for k in mapped)})\n"

    if name == "drag_and_drop":
        dest_x = _denormalize(args.get("destination_x", 0), screen_w)
        dest_y = _denormalize(args.get("destination_y", 0), screen_h)
        return f"pyautogui.moveTo({x}, {y})\npyautogui.dragTo({dest_x}, {dest_y}, duration=0.5)\n"

    wait_match = re.fullmatch(r"wait_(\d+)_seconds", name)
    if wait_match:
        seconds = max(1, min(int(wait_match.group(1)), 30))
        return f"time.sleep({seconds})\n"

    # Browser actions that we handle as no-ops or best-effort
    if name in _EXCLUDED_ACTIONS:
        logger.warning("Gemini CUA returned browser action '%s' — skipping", name)
        return "time.sleep(0.1)\n"

    logger.warning("Unknown Gemini CUA action: %s (args=%s)", name, args)
    return "time.sleep(0.1)\n"


class GeminiCUAAgent:
    """Gemini computer-use agent using the google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        screen_size: tuple[int, int] = (1920, 1080),
        password: str = "",
    ):
        from google import genai
        from google.genai import types

        self.model = model
        self.screen_size = screen_size
        self.password = password
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._types = types
        self._contents: list = []
        self._pending_calls: list[tuple] = []
        self._step = 0
        self._last_usage: dict = {}

    def reset(self) -> None:
        """Reset agent state."""
        self._contents = []
        self._pending_calls = []
        self._step = 0

    def _prune_old_screenshots(self) -> None:
        """Keep visual context bounded, matching Google's reference agent loop."""
        turns_with_screenshots = 0
        for content in reversed(self._contents):
            if getattr(content, "role", None) != "user" or not getattr(content, "parts", None):
                continue

            has_screenshot = False
            for part in content.parts:
                function_response = getattr(part, "function_response", None)
                if (
                    function_response
                    and getattr(function_response, "parts", None)
                    and getattr(function_response, "name", None) in _PREDEFINED_COMPUTER_USE_FUNCTIONS
                ):
                    has_screenshot = True
                    continue
                if getattr(part, "inline_data", None):
                    has_screenshot = True

            if not has_screenshot:
                continue

            turns_with_screenshots += 1
            if turns_with_screenshots <= MAX_RECENT_TURN_WITH_SCREENSHOTS:
                continue

            retained_parts = []
            for part in content.parts:
                function_response = getattr(part, "function_response", None)
                if (
                    function_response
                    and getattr(function_response, "parts", None)
                    and getattr(function_response, "name", None) in _PREDEFINED_COMPUTER_USE_FUNCTIONS
                ):
                    function_response.parts = None
                    retained_parts.append(part)
                elif getattr(part, "inline_data", None):
                    continue
                else:
                    retained_parts.append(part)
            content.parts = retained_parts

    def predict(self, instruction: str, obs: dict, **kwargs) -> tuple[str, list[str]]:
        """Return (raw_response, [pyautogui_code_or_signal])."""
        types = self._types
        from google.genai.types import Content, FunctionResponse, FunctionResponseBlob, FunctionResponsePart, Part

        config = types.GenerateContentConfig(
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=_EXCLUDED_ACTIONS,
                    )
                )
            ],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=(
                SYSTEM_PROMPT_TEMPLATE.replace("{DATE}", time.strftime("%A, %B %d, %Y")).replace(
                    "{PASSWORD}", self.password
                )
                + "\n\n"
                + GEMINI_DESKTOP_NOTE
            ),
        )

        screenshot_bytes = obs["screenshot"]
        media_type = obs.get("media_type", "image/jpeg")
        current_url = obs.get("current_url") or "about:blank"
        prev_exec_results = kwargs.get("prev_exec_results") or []

        from PIL import Image

        try:
            img = Image.open(io.BytesIO(screenshot_bytes))
            observed_screen_size = img.size
        except Exception:
            img = None
            observed_screen_size = self.screen_size

        # Gemini computer-use requires PNG screenshots
        if media_type != "image/png":
            if img is None:
                img = Image.open(io.BytesIO(screenshot_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            screenshot_bytes = buf.getvalue()
            media_type = "image/png"

        screen_w, screen_h = observed_screen_size
        self.screen_size = (int(screen_w), int(screen_h))

        if self._step == 0:
            # Initial request: instruction + screenshot
            self._contents = [
                Content(
                    role="user",
                    parts=[
                        Part(text=instruction),
                        Part.from_bytes(data=screenshot_bytes, mime_type=media_type),
                    ],
                )
            ]
        else:
            # Feed back screenshot as function responses for each pending call
            if self._pending_calls:
                parts = []
                for index, (fc, safety_ack_required) in enumerate(self._pending_calls):
                    exec_result = prev_exec_results[index] if index < len(prev_exec_results) else {}
                    response_payload = {"url": exec_result.get("url") or current_url}
                    if exec_result.get("returncode") not in (None, 0):
                        response_payload["error"] = exec_result.get("error") or "action failed"
                    if safety_ack_required:
                        response_payload["safety_acknowledgement"] = "true"
                    fr = FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response=response_payload,
                        parts=[
                            FunctionResponsePart(
                                inline_data=FunctionResponseBlob(
                                    mime_type=media_type,
                                    data=screenshot_bytes,
                                )
                            )
                        ],
                    )
                    parts.append(Part(function_response=fr))
                self._contents.append(Content(role="user", parts=parts))

        self._prune_old_screenshots()

        # API call with retry (only retry on transient errors, not 400s)
        response = None
        for attempt in range(API_RETRY_TIMES):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=self._contents,
                    config=config,
                )
                break
            except Exception as e:
                err_str = str(e)
                is_client_error = "400" in err_str or "INVALID_ARGUMENT" in err_str
                if is_client_error:
                    logger.error("Gemini API client error (not retryable): %s", err_str[:500])
                    return err_str, ["FAIL"]
                logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, API_RETRY_TIMES, err_str[:300])
                if attempt < API_RETRY_TIMES - 1:
                    time.sleep(API_RETRY_INTERVAL)
                else:
                    logger.error("All Gemini API attempts failed")
                    return err_str, ["FAIL"]

        if response is None:
            return "No response", ["FAIL"]

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt_tokens = _usage_int(usage_meta, "prompt_token_count")
            cached_tokens = _usage_int(usage_meta, "cached_content_token_count")
            tool_prompt_tokens = _usage_int(usage_meta, "tool_use_prompt_token_count")
            candidate_tokens = _usage_int(usage_meta, "candidates_token_count")
            thought_tokens = _usage_int(usage_meta, "thoughts_token_count")
            total_tokens = _usage_int(usage_meta, "total_token_count")
            self._last_usage = {
                # Gemini reports total_token_count as prompt + tool-use prompt +
                # candidates + thoughts. Keep the same accounting shape as the
                # other CUA adapters: fresh input excludes cached reads, and
                # output includes hidden thinking tokens when present.
                "input_tokens": max(prompt_tokens - cached_tokens, 0) + tool_prompt_tokens,
                "output_tokens": candidate_tokens + thought_tokens,
                "cache_read_input_tokens": cached_tokens,
                "gemini_prompt_token_count": prompt_tokens,
                "gemini_tool_use_prompt_token_count": tool_prompt_tokens,
                "gemini_candidates_token_count": candidate_tokens,
                "gemini_thoughts_token_count": thought_tokens,
                "gemini_total_token_count": total_tokens,
            }
        else:
            self._last_usage = {}

        # Append model response to conversation
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        if content is None or content.parts is None:
            self._step += 1
            self._pending_calls = []
            finish = candidate.finish_reason if candidate else "unknown"
            return f"[No content from model, finish_reason={finish}]", ["WAIT"]

        self._contents.append(content)
        self._step += 1

        # Extract function calls and text
        raw_parts: list[str] = []
        actions: list[str] = []
        self._pending_calls = []

        for part in content.parts:
            if part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                safety = args.get("safety_decision")
                safety_decision = (
                    safety.get("decision") if isinstance(safety, dict) else getattr(safety, "decision", None)
                )
                safety_explanation = (
                    safety.get("explanation") if isinstance(safety, dict) else getattr(safety, "explanation", "")
                )
                safety_ack_required = safety_decision == "require_confirmation"
                if safety_ack_required:
                    logger.info(
                        "Safety confirmation requested: %s — acknowledging in next FunctionResponse",
                        safety_explanation,
                    )
                self._pending_calls.append((fc, safety_ack_required))
                code = _action_to_pyautogui(fc.name, args, screen_w, screen_h)
                actions.append(code)
                raw_parts.append(f"[ACTION] {fc.name}({args})")
            elif part.text:
                raw_parts.append(f"[TEXT] {part.text}")
                if "INFEASIBLE" in part.text:
                    return "\n".join(raw_parts), ["FAIL"]

        raw_str = "\n".join(raw_parts)

        if not actions:
            actions = ["DONE"]

        return raw_str, actions
