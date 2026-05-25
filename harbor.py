"""Harbor BaseAgent wrapper for Cifrato CUA agents."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from harbor.agents.base import AgentContext, BaseAgent

from .anthropic_cua import AnthropicCUAAgent
from .gemini_cua import GeminiCUAAgent
from .openai_cua import OpenAICUAAgent
from env.agents.utils.pricing import estimate_cost
from env.common import ensure_dotenv
from env.harbor.utils.trajectory import TrajectoryBuilder
from env.harbor.utils.verifier import run_evaluation_use_computer

logger = logging.getLogger(__name__)

DEFAULT_MODELS: dict[str, str | list[str]] = {
    "openai_cua": "gpt-5.4-2026-03-05",
    "anthropic_cua": "claude-sonnet-4-6",
    "gemini_cua": "gemini-2.5-computer-use-preview-10-2025",
}

_AGENT_FACTORIES = {
    "anthropic_cua": lambda m, **kw: AnthropicCUAAgent(
        model=m,
        max_tokens=kw.get("max_tokens", 4096),
        temperature=kw.get("temperature"),
        top_p=kw.get("top_p"),
        screen_size=(kw.get("screen_width", 1920), kw.get("screen_height", 1080)),
        password=kw.get("password", ""),
    ),
    "openai_cua": lambda m, **kw: OpenAICUAAgent(
        model=m,
        screen_size=(kw.get("screen_width", 1920), kw.get("screen_height", 1080)),
        password=kw.get("password", ""),
    ),
    "gemini_cua": lambda m, **kw: GeminiCUAAgent(
        model=m,
        screen_size=(kw.get("screen_width", 1920), kw.get("screen_height", 1080)),
        password=kw.get("password", ""),
    ),
}


class CifratoCUA(BaseAgent):
    """Harbor agent wrapping Cifrato's screenshot-loop CUA agents."""

    SUPPORTS_ATIF = True
    SUPPORTS_WINDOWS = True

    def __init__(
        self,
        logs_dir: Path | None = None,
        model_name: str = "anthropic/claude-sonnet-4-6",
        agent_type: str = "anthropic_cua",
        max_steps: int = 15,
        pause: float = 2.0,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        password: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
        evaluate: bool = True,
        **kwargs: Any,
    ):
        if logs_dir is None:
            logs_dir = Path(tempfile.mkdtemp(prefix="cifrato-cua-"))
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._agent_type = agent_type
        self._max_steps = max_steps
        self._pause = pause
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._password = password
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._evaluate = evaluate
        self._task_config_path: str | None = None

    def name(self) -> str:
        return f"cifrato-{self._agent_type}"

    def version(self) -> str | None:
        return "1.0.0"

    @property
    def _model(self) -> str:
        if "/" in self.model_name:
            return self.model_name.split("/", 1)[1]
        return self.model_name

    def _find_task_config(self, environment) -> None:
        """Locate task_config.json from the task directory."""
        env_dir = getattr(environment, "environment_dir", None)
        if env_dir:
            config_path = Path(env_dir).parent / "tests" / "task_config.json"
            if config_path.exists():
                self._task_config_path = str(config_path)
                return
        task_dir = getattr(environment, "_task_dir", None)
        if task_dir:
            config_path = Path(task_dir) / "tests" / "task_config.json"
            if config_path.exists():
                self._task_config_path = str(config_path)

    async def setup(self, environment) -> None:
        ensure_dotenv()
        self._find_task_config(environment)
        if not self._task_config_path:
            logger.warning("No task_config.json found — skipping setup")
            return
        if not getattr(environment, "is_use_computer_windows", False):
            raise RuntimeError("CifratoCUA only supports CifratoUseComputerWindows")
        logger.info("use.computer Windows setup is handled by the environment")

    async def run(self, instruction: str, environment, context: AgentContext) -> None:
        ensure_dotenv()
        if not getattr(environment, "is_use_computer_windows", False):
            raise RuntimeError("CifratoCUA only supports CifratoUseComputerWindows")

        factory = _AGENT_FACTORIES.get(self._agent_type)
        if not factory:
            raise ValueError(f"Unknown agent type: {self._agent_type}")

        agent = factory(
            self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            password=self._password,
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            max_steps=self._max_steps,
        )
        agent.reset()

        await self._start_recording(environment)

        traj = TrajectoryBuilder(self.name(), self.version(), self._model, instruction=instruction)
        total_input = total_output = total_cache_create = total_cache_read = 0
        total_cost = 0.0
        step = 0
        done = False
        timed_out = False
        prev_exec_results: list[dict] = []

        try:
            while not done and step < self._max_steps:
                try:
                    screenshot_bytes = await self._screenshot(environment)
                except Exception as sc_err:
                    logger.warning("Screenshot failed: %s — retrying once after 5s", sc_err)
                    await asyncio.sleep(5)
                    screenshot_bytes = await self._screenshot(environment)
                (self.logs_dir / f"step_{step + 1}.jpg").write_bytes(screenshot_bytes)

                obs = {
                    "screenshot": screenshot_bytes,
                    "media_type": "image/jpeg",
                    "current_url": await self._current_url(environment),
                }
                response_text, actions = agent.predict(instruction, obs, prev_exec_results=prev_exec_results)

                # Track usage — include cache tokens in the input total for accurate display
                usage = getattr(agent, "_last_usage", {}) or {}
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                step_in = usage.get("input_tokens", 0) + cache_create + cache_read
                step_out = usage.get("output_tokens", 0)
                total_input += step_in
                total_output += step_out
                total_cache_create += cache_create
                total_cache_read += cache_read
                total_cost += estimate_cost(self._model, usage) if usage else 0

                logger.info("[Step %d] %s", step + 1, response_text[:300])

                # Execute actions and build trajectory data
                tool_calls, observations, exec_results = [], [], []
                for action in actions:
                    call_id = str(uuid.uuid4())[:8]

                    if action in ("DONE", "FAIL"):
                        logger.info("Agent signaled %s.", action)
                        tool_calls.append(
                            {"tool_call_id": call_id, "tool_name": "signal", "parameters": {"signal": action}}
                        )
                        observations.append({"source_call_id": call_id, "content": f"Agent signaled {action}."})
                        done = True
                        break
                    elif action == "WAIT":
                        tool_calls.append(
                            {"tool_call_id": call_id, "tool_name": "wait", "parameters": {"seconds": self._pause}}
                        )
                        observations.append({"source_call_id": call_id, "content": f"Waited {self._pause}s."})
                        await asyncio.sleep(self._pause)
                    elif action != "unknown":
                        tool_calls.append(
                            {"tool_call_id": call_id, "tool_name": "pyautogui", "parameters": {"code": action}}
                        )
                        try:
                            result = await self._execute_action(environment, action)
                        except Exception as exec_err:
                            logger.warning("Action execution failed: %s", exec_err)
                            result = {"returncode": 1, "output": "", "error": str(exec_err)[:500]}
                        exec_results.append(
                            {
                                "action": action,
                                "returncode": result.get("returncode"),
                                "output": (result.get("output", "") or "")[:1500],
                                "error": (result.get("error", "") or "")[:1500],
                                "url": result.get("url", ""),
                            }
                        )
                        obs_text = result.get("output", "") or ""
                        if result.get("returncode", 0) != 0:
                            obs_text = f"ERROR (rc={result.get('returncode')}): {result.get('error', '')[:300]}"
                            logger.warning("Action failed: %s", obs_text[:300])
                        observations.append({"source_call_id": call_id, "content": obs_text[:1500] or "(no output)"})
                        await asyncio.sleep(self._pause)

                traj.add_step(
                    screenshot_bytes=screenshot_bytes,
                    response_text=response_text,
                    tool_calls=tool_calls or None,
                    observations=observations or None,
                    input_tokens=step_in or None,
                    output_tokens=step_out or None,
                )
                prev_exec_results = exec_results
                step += 1

        except (asyncio.CancelledError, TimeoutError):
            logger.warning("Agent timed out after %d steps — saving partial trajectory", step)
            timed_out = True
        finally:
            await self._stop_recording_safely(environment, timed_out=timed_out)

        # Populate context
        if done:
            result_str = "DONE"
        elif timed_out:
            result_str = "timeout"
        else:
            result_str = "max_steps"
        context.n_input_tokens = total_input
        context.n_cache_tokens = total_cache_read or None
        context.n_output_tokens = total_output
        context.cost_usd = total_cost or None
        context.metadata = {
            "agent_type": self._agent_type,
            "model": self._model,
            "total_steps": step,
            "result": result_str,
            "estimated_cost_usd": round(total_cost, 4),
        }
        if total_cache_create:
            context.metadata["cache_creation_tokens"] = total_cache_create
        if total_cache_read:
            context.metadata["cache_read_tokens"] = total_cache_read

        # Evaluate (skip on timeout — VM state may be mid-action)
        eval_score = None
        if self._evaluate and self._task_config_path and not timed_out:
            score, artifacts = await run_evaluation_use_computer(
                environment,
                self._task_config_path,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )
            eval_score = score
            context.metadata["eval_score"] = score
            context.metadata["eval_artifacts"] = artifacts

            # Write reward.txt + test-stdout.txt so Harbor's verifier/viewer picks it up
            verifier_dir = self.logs_dir.parent / "verifier"
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "reward.txt").write_text(str(score))
            # Format getter results as human-readable stdout
            stdout_lines = [f"Score: {score}"]
            for g in artifacts.get("getter_results", []):
                if g.get("error"):
                    stdout_lines.append(f"ERROR: {g['error']}")
                elif g.get("result"):
                    stdout_lines.append(g["result"])
            (verifier_dir / "test-stdout.txt").write_text("\n".join(stdout_lines))

        # Write trajectory (always — even partial on timeout)
        traj.write(
            self.logs_dir,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=total_cost,
            reward=eval_score,
        )

        # Re-raise so Harbor still records the timeout
        if timed_out:
            raise asyncio.CancelledError()

    async def _screenshot(self, environment) -> bytes:
        return await environment.screenshot()

    async def _current_url(self, environment) -> str:
        if not hasattr(environment, "current_url"):
            return ""
        try:
            return await environment.current_url()
        except Exception as exc:
            logger.debug("Could not read current browser URL: %s", exc)
            return ""

    async def _execute_action(self, environment, action: str) -> dict:
        return await environment.execute_pyautogui(action)

    async def _start_recording(self, environment) -> None:
        await environment.start_recording()

    async def _stop_recording(self, environment, *, timed_out: bool) -> None:
        recording_path = self.logs_dir / "recording.mp4"
        rec_timeout = 15 if timed_out else 120
        await environment.stop_recording(recording_path, timeout=rec_timeout)

    async def _stop_recording_safely(self, environment, *, timed_out: bool) -> None:
        try:
            await asyncio.shield(self._stop_recording(environment, timed_out=timed_out))
        except asyncio.CancelledError:
            logger.warning("Cancelled while saving recording; relying on environment teardown fallback")
        except Exception as e:
            logger.warning("Could not save recording: %s", e)
