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

from openai import OpenAI

from models import (
    SatelliteSchedulerAction,
    ActionType,
    SatelliteSchedulerObservation,
)
from client import SatelliteSchedulerEnv
from grader import grade_all


IMAGE_NAME = os.getenv("IMAGE_NAME", "openenv-satellite_scheduler:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("SATELLITE_SCHEDULER_TASK", "satellite_mission_planning")

TASK_EASY = os.getenv(
    "SATELLITE_SCHEDULER_TASK_EASY", "priority_aware_mission_planning"
)
TASK_MEDIUM = os.getenv("SATELLITE_SCHEDULER_TASK_MEDIUM", "storage_flow_efficiency")
TASK_HARD = os.getenv("SATELLITE_SCHEDULER_TASK_HARD", "closed_loop_mission_planning")

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
    obs: SatelliteSchedulerObservation,
    last_reward: float,
    history: List[str],
) -> str:
    # Extract all state attributes from observation
    current_time = obs.current_time
    attitude = obs.attitude
    busy_status = obs.busy_status
    remaining_action_steps = obs.remaining_action_steps
    battery_level = obs.battery_level
    storage_used = obs.storage_used
    raw_data_amount = obs.raw_data_amount
    compressed_data_amount = obs.compressed_data_amount
    sunlit_status = obs.sunlit_status
    ground_station_visible = obs.ground_station_visible
    pending_request_queue = obs.pending_request_queue
    current_selected_request_id = obs.current_selected_request_id

    sunlit_str = "SUNLIT" if sunlit_status else "DARK"
    gs_str = "VISIBLE" if ground_station_visible else "NOT VISIBLE"

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
        queue_str = "(error reading queue)"

    return textwrap.dedent(
        f"""
        SATELLITE STATE
        ===============
        Step: {step}/{MAX_STEPS}
        
        ORBITAL STATE
        Time: {current_time}s | Attitude: {attitude} | Busy Status: {busy_status}
        Remaining Action Steps: {remaining_action_steps}
        
        RESOURCE LEVELS
        Battery: {battery_level:.1f}% | Storage Used: {storage_used:.1f} GB
        Raw Data: {raw_data_amount:.1f} GB | Compressed Data: {compressed_data_amount:.1f} GB
        
        VISIBILITY
        Sunlit: {sunlit_str} | Ground Station: {gs_str}
        
        MISSION REQUESTS
        {queue_str}
        Currently Selected Request: {current_selected_request_id if current_selected_request_id else "None"}
        
        PERFORMANCE
        Last Reward: {last_reward:.3f}
        
        RECENT ACTIONS
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
        f"[END] success={str(success).lower()} steps={steps} score={score} rewards={rewards_str}",
        flush=True,
    )


def get_model_decision(
    client: OpenAI,
    step: int,
    obs: SatelliteSchedulerObservation,
    last_reward: float,
    history: List[str],
) -> str:
    try:
        user_prompt = build_user_prompt(
            step,
            obs,
            last_reward,
            history,
        )
    except Exception as e:
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
    except Exception as e:
        return "wait"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_configs = [
        (TASK_EASY, "easy"),
        (TASK_MEDIUM, "medium"),
        (TASK_HARD, "hard"),
    ]

    for task_name, grade_key in task_configs:
        env = await SatelliteSchedulerEnv.from_docker_image(IMAGE_NAME)

        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset()

            last_reward = 0.01

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs = result.observation

                decision = get_model_decision(
                    client,
                    step,
                    obs,
                    last_reward,
                    history,
                )

                action = parse_action_from_text(decision)

                result = await env.step(action)

                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None

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

            # Evaluate each run against the corresponding grader.
            if steps_taken >= MAX_STEPS or (result and result.done):
                state = await env.state()
                episode_stats = state.episode_stats
                grades = grade_all(episode_stats)
                score = grades.get(grade_key, 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
        except Exception:
            pass
        finally:
            try:
                await env.close()
            except Exception:
                # Docker stop timed out, attempt force kill.
                try:
                    if hasattr(env, "_container_id"):
                        subprocess.run(
                            ["docker", "kill", env._container_id],
                            timeout=5,
                            capture_output=True,
                        )
                except Exception:
                    pass

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    sys.tracebacklimit = 1
    asyncio.run(main())
