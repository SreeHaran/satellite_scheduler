"""
Satellite Scheduler Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import re
import subprocess
import sys
import textwrap
import traceback
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError as e:
    # print(f"[DEBUG] Failed to import openai: {e}", flush=True)
    raise

try:
    from models import SatelliteSchedulerAction, ActionType, TargetRequest
except ImportError as e:
    # print(f"[DEBUG] Failed to import models: {e}", flush=True)
    raise

try:
    from client import SatelliteSchedulerEnv
except ImportError as e:
    # print(f"[DEBUG] Failed to import client: {e}", flush=True)
    raise

try:
    from grader import grade_all
except ImportError as e:
    # print(f"[DEBUG] Failed to import grader: {e}", flush=True)
    raise

IMAGE_NAME = os.getenv("IMAGE_NAME", "openenv-satellite_scheduler:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("SATELLITE_SCHEDULER_TASK", "satellite_mission_planning")
BENCHMARK = os.getenv("SATELLITE_SCHEDULER_BENCHMARK", "satellite_scheduler")

MAX_STEPS = 50  # Reduced to meet hackathon requirements of execution within 20 minutes(in vcpu=2, memory=8gb), Its best to run with 180 steps
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent controlling a satellite for mission-critical imaging and data downlink.
    Your goal: Capture priority targets, compress data, downlink to ground station, and manage battery.
    
    Available actions: wait, abort_task, sun_point_for_charging, compress_data, downlink_to_station, capture_image
    
    Strategy tips:
    - Prioritize high-priority targets
    - Charge during sunlit windows when not imaging
    - Compress captured data before downlink
    - Downlink when ground station is visible (gs_visible=true)
    - Monitor battery and storage levels to avoid failure
    
    Respond with a single action name and optional parameters. Note that optional parameters (like target_id) should only be included for the capture_image action.
    The target id for the capture_image action should be selected from the pending request queue, prioritizing higher priority targets.
    Output format:
    - For actions without parameters: "wait", "abort_task", "sun_point_for_charging", "compress_data", "downlink_to_station"
    - For capture_image action: "capture_image <target-id>". Example: "capture_image 42"
    """
).strip()


def build_user_prompt(
    step: int,
    current_time: float,
    battery: float,
    storage_util: float,
    is_sunlit: bool,
    gs_visible: bool,
    pending_request_queue: List[TargetRequest],
    last_reward: float,
    history: List[str],
) -> str:
    sunlit_str = "SUNLIT" if is_sunlit else "DARK"
    gs_str = "VISIBLE" if gs_visible else "NOT VISIBLE"

    try:
        if pending_request_queue:
            queue_lines = "\n".join(
                f"  - id={r.request_id} priority={r.priority.value} mode={r.imaging_mode.value} deadline={r.deadline}s"
                for r in pending_request_queue
            )
            queue_str = f"{len(pending_request_queue)} pending:\n{queue_lines}"
        else:
            queue_str = "none"
    except Exception as e:
        # print(
        #     f"[DEBUG] build_user_prompt: error building queue string: {e}",
        #     flush=True,
        # )
        queue_str = "(error reading queue)"

    return textwrap.dedent(
        f"""
        MISSION STATUS
        ==============
        Step: {step}/{MAX_STEPS} (Time: {current_time:.0f}s)
        Battery: {battery:.1f}% | Storage: {storage_util:.0f}% | Sunlit: {sunlit_str} | GS: {gs_str}
        Requests: {queue_str} | Last reward: {last_reward:.3f}
        
        Recent actions:
        {chr(10).join(history[-3:]) if history else "(none)"}
        
        What action should the satellite take next?
    """
    ).strip()


def parse_action_from_text(text: str) -> Optional[SatelliteSchedulerAction]:
    """Parse LLM response into a SatelliteSchedulerAction."""
    try:
        text_lower = text.lower().strip()

        # Handle capture_image first: extract target_id from response (e.g. "capture_image 42")
        if "capture" in text_lower or "image" in text_lower:
            match = re.search(r"\d+", text)
            target_id = int(match.group()) if match else None
            return SatelliteSchedulerAction(
                action_type=ActionType.CAPTURE_IMAGE, target_id=target_id
            )

        action_mappings = {
            "wait": ActionType.WAIT,
            "abort": ActionType.ABORT_TASK,
            "sun_point": ActionType.SUN_POINT_FOR_CHARGING,
            "charge": ActionType.SUN_POINT_FOR_CHARGING,
            "compress": ActionType.COMPRESS_DATA,
            "downlink": ActionType.DOWNLINK_TO_STATION,
        }

        for keyword, action_type in action_mappings.items():
            if keyword in text_lower:
                return SatelliteSchedulerAction(action_type=action_type)

        # Default to wait if unclear
        return SatelliteSchedulerAction(action_type=ActionType.WAIT)
    except Exception as e:
        # print(
        #     f"[DEBUG] parse_action_from_text failed: {e}",
        #     flush=True,
        # )
        return SatelliteSchedulerAction(action_type=ActionType.WAIT)


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def get_model_decision(
    client: OpenAI,
    step: int,
    current_time: float,
    battery: float,
    storage_util: float,
    is_sunlit: bool,
    gs_visible: bool,
    pending_request_queue: List[TargetRequest],
    last_reward: float,
    history: List[str],
) -> str:
    try:
        user_prompt = build_user_prompt(
            step,
            current_time,
            battery,
            storage_util,
            is_sunlit,
            gs_visible,
            pending_request_queue,
            last_reward,
            history,
        )
    except Exception as e:
        # print(
        #     f"[DEBUG] get_model_decision: build_user_prompt failed at step={step}: {e}",
        #     flush=True,
        # )
        return "wait"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "wait"
    except Exception as exc:
        # print(
        #     f"[DEBUG] Model request failed at step={step}: {exc}",
        #     flush=True,
        # )
        return "wait"


async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        raise Exception(f"Failed to create OpenAI client: {e}")

    try:
        env = await SatelliteSchedulerEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        raise Exception(f"from_docker_image failed (image={IMAGE_NAME}): {e}")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            result = await env.reset()
        except Exception as e:
            raise Exception(f"env.reset() failed: {e}")

        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                obs = result.observation
            except Exception as e:
                raise Exception(f"Failed to read observation at step={step}: {e}")

            decision = get_model_decision(
                client,
                step,
                obs.current_time,
                obs.battery_level,
                obs.storage_used,
                obs.sunlit_status,
                obs.ground_station_visible,
                obs.pending_request_queue,
                last_reward,
                history,
            )

            action = parse_action_from_text(decision)

            try:
                result = await env.step(action)
            except Exception as e:
                raise Exception(
                    f"env.step() failed at step={step} action={action.action_type.name}: {e}"
                )

            try:
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as e:
                raise Exception(f"Failed to unpack step result at step={step}: {e}")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(
                step=step,
                action=f"{action.action_type.name}",
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action.action_type.name} (reward={reward:+.3f})"
            )

            if done:
                break

        # Use episode grade as final score
        if steps_taken >= MAX_STEPS or (result and result.done):
            try:
                state = await env.state()
                episode_stats = state.episode_stats
                grades = grade_all(episode_stats)
                score = (grades["easy"] + grades["medium"] + grades["hard"]) / 3.0
                success = score >= SUCCESS_SCORE_THRESHOLD
                # print(
                #     f"[DEBUG] Episode grades - Easy: {grades['easy']:.3f}, Medium: {grades['medium']:.3f}, Hard: {grades['hard']:.3f}",
                #     flush=True,
                # )
            except Exception as e:
                raise Exception(f"Could not compute grades: {e}")

    except Exception as e:
        raise Exception(f"Unhandled exception in episode loop: {e}")
    finally:
        try:
            await env.close()
        except Exception as e:
            # Docker stop timed out, attempt force kill
            try:
                if hasattr(env, "_container_id"):
                    subprocess.run(
                        ["docker", "kill", env._container_id],
                        timeout=5,
                        capture_output=True,
                    )
            except Exception as e:
                raise Exception(f"env.close() failed (container cleanup): {e}")

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    sys.tracebacklimit = 1
    asyncio.run(main())
