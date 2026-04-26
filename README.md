---
title: Satellite Scheduler Environment Server
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Satellite Scheduler Environment

An RL environment simulating a low-Earth-orbit (LEO) satellite that must schedule imaging tasks, manage battery via sun-pointing, compress captured data, and downlink results during ground-station visibility windows — all within a 90-minute orbital episode.

[Blog link](Blog.md)

[Space link](https://huggingface.co/spaces/Sreeharan/satellite_scheduler)

[Github link](https://github.com/SreeHaran/satellite_scheduler)

[SFT training notebook link (also attached to this repo)](training/sft_training.ipynb)

[GRPO training notebook link (also attached to this repo)](training/grpo_training.ipynb)


## Problem Statement

Real LEO Earth-observation satellites face a hard resource-scheduling problem: they must serve a continuously arriving stream of ground imaging requests while simultaneously managing three scarce, time-coupled resources — **battery**, **onboard storage**, and **ground-station contact time**.

The constraints interact in non-trivial ways:

- **Charging only works in sunlight.** Every orbit has fixed eclipse periods where the solar panels produce nothing, yet every action drains the battery.
- **Storage is finite.** Raw captured images consume 5–15 GB each. If storage fills up, no new images can be taken and the mission stalls.
- **Downlink windows are short and infrequent.** Ground stations are only visible for brief windows (~10 minutes per pass). Data must be *compressed* before it can be downlinked, so the pipeline `capture → compress → downlink` must be choreographed across multiple orbital windows.
- **Requests have deadlines and priorities.** Imaging opportunities expire if the satellite does not act in time. High-priority targets (e.g., disaster monitoring) carry more value than routine surveys.

An onboard scheduler today is typically a hand-crafted rule engine. This environment frames the problem as a **sequential decision-making task** so a reinforcement-learning agent can learn a policy that maximises mission value end-to-end, including trade-offs that are hard to enumerate manually (e.g., "should I charge now or rush to capture this high-priority target before its deadline?").

## How the RL Agent Solves It

At each 30-second step the agent receives a full observation of the satellite's state and picks one of six discrete actions. Multi-step actions (capture, compress, downlink, charge) run over several consecutive steps — the agent only needs to *start* them; the environment continues them automatically until the agent sends `abort_task` or the action naturally completes.

The agent must learn to:

1. **Prioritise requests** — prefer high-priority captures and respect deadlines.
2. **Pipeline data** — never let raw data sit idle when storage is tight; compress promptly.
3. **Time the downlink** — buffer compressed data and flush it during every GS pass.
4. **Manage energy** — charge proactively during sunlit windows before battery drops critical.
5. **Recover from mistakes** — use `abort_task` when a better opportunity appears mid-action.



## Graders

Performance is evaluated by three graders of increasing difficulty. All graders read `episode_stats` returned by `GET /state` at the end of an episode and return a score in **[0, 1]** (higher is better).

---

### Grader 1 — Easy: Priority-Aware Mission Completion

> *Did the agent capture the most valuable targets?*

```
score = Σ priority_weight(completed requests)
        ─────────────────────────────────────
        Σ priority_weight(all requests)
```

| Priority | Weight |
|---|---|
| low | 1.0 |
| medium | 2.0 |
| high | 3.0 |

A request counts as "completed" once its status reaches `captured`, `compressed`, or `downlinked`. This grader ignores whether data was actually sent to the ground — it rewards the agent purely for acquiring the right images.

**Perfect score:** capture every request, highest-priority ones first.

---

### Grader 2 — Medium: Storage-Flow Efficiency

> *Did the agent keep data moving through the pipeline without clogging storage?*

```
score = 0.50 × (data_downlinked  / data_compressed)   ← downlink throughput
      + 0.30 × (data_compressed  / data_generated)    ← compression rate
      + 0.20 × (1 − avg_storage_utilisation)          ← storage headroom
      − 0.15 × stalled_raw_data_fraction              ← idle raw data penalty
      − 0.10 × storage_full_steps_fraction            ← high-storage penalty
```

This grader penalises the agent for capturing data it cannot process or transmit. An agent that fills storage and then does nothing loses points even if it captured many requests.

**Perfect score:** compress all raw data immediately and downlink every byte during GS passes.

---

### Grader 3 — Hard: Closed-Loop Mission Planning

> *Did the agent manage all subsystems end-to-end without failures?*

```
score = 0.50 × priority_weighted_completion_ratio
      + 0.30 × (data_downlinked / data_generated)     ← end-to-end pipeline
      + 0.10 × charging_efficiency                    ← used sunlit windows well
      + 0.10 × storage_efficiency                     ← avoided overflow events
      − 0.15 × missed_deadlines_fraction              ← requests that expired
      − 0.20 × battery_low_fraction                   ← steps below 10 % battery
      − 0.20 × storage_high_fraction                  ← steps above 85 GB storage
```

Where:
- **charging_efficiency** = `min(1, total_battery_gained / (sunlit_steps × 10))` — did the agent use every available solar window?
- **storage_efficiency** = `1 − (overflow_events + high_storage_steps) / total_steps` — did storage stay safely below capacity?

This grader can return slightly negative scores for catastrophically managed episodes (battery death, perpetual storage overflow). It is the definitive measure of an agent that has mastered all four sub-tasks simultaneously.

**Perfect score:** capture all high-priority requests before deadlines, keep the pipeline flowing, never let battery drop below 10 %, and downlink everything.

## Quick Start

```python
from satellite_scheduler import (
    SatelliteSchedulerAction,
    SatelliteSchedulerEnv,
    ActionType,
)

with SatelliteSchedulerEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(f"Battery: {result.observation.battery_level}")
    print(f"Requests: {len(result.observation.pending_request_queue)}")

    # Wait one step
    result = env.step(SatelliteSchedulerAction(action_type=ActionType.WAIT))
    print(f"Time: {result.observation.current_time}s")

    # Slew to target 1
    result = env.step(
        SatelliteSchedulerAction(action_type=ActionType.WAIT)
    )
    print(f"Attitude: {result.observation.attitude}")
```

## Repository Structure

```
satellite_scheduler/
├── client.py                    # SatelliteSchedulerEnv — async WebSocket client for the environment
├── models.py                    # Pydantic data models (Action, Observation, ActionType, TargetRequest, etc.)
├── grader.py                    # Three grading functions: grade_easy, grade_medium, grade_hard, grade_all
├── inference.py                 # Reference inference loop — runs a full episode with an OpenAI-compatible LLM
├── validate-submission.sh       # End-to-end validation script for submissions
├── Dockerfile                   # Docker image for the environment server
├── openenv.yaml                 # OpenEnv deployment config
├── pyproject.toml               # Python project metadata and dependencies
│
├── server/                      # Environment server (runs the simulation)
│   ├── app.py                   # FastAPI + WebSocket entry point
│   ├── satellite_scheduler_environment.py  # Core simulation logic (orbit, resources, step function)
│   ├── orbit.py                 # Orbital mechanics — sunlit/eclipse windows, ground-station visibility
│   ├── metrics.py               # Episode statistics accumulator for graders
│   ├── requirements.txt         # Server-specific dependencies
│   └── actions/                 # One module per action type
│       ├── capture.py           # capture_image — slew + image acquisition
│       ├── compress.py          # compress_data — raw → compressed conversion
│       ├── downlink.py          # downlink_to_station — transmit during GS pass
│       ├── sun_point.py         # sun_point_for_charging — solar panel charging
│       └── wait_abort.py        # wait + abort_task handlers
│
├── dataset/                     # Training data (generated from expert rollouts)
│   ├── sft_training.jsonl       # 5,000 (system, prompt, completion) examples for SFT
│   ├── sft_validation.jsonl     # Validation split for SFT
│   └── grpo_training.jsonl      # Prompt + gold-action pairs for GRPO reward functions
│
├── training/                    # Training notebooks (run on Kaggle)
│   ├── sft_training.ipynb       # Supervised fine-tuning notebook
│   └── grpo-training.ipynb      # GRPO reinforcement learning notebook
│
├── outputs/                     # Trained model artifacts - ignored by git
│   ├── lora_adapter/            # LoRA adapter weights (safetensors + tokenizer)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── chat_template.jinja
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── sft_training_loss.png    # SFT training loss curve
│
└── assets/                      # Images and diagrams for documentation
    ├── actions.png
    ├── state variables.png
    ├── grpo_on_full_episode/            # Full episode visualizations
    ├── grpo_data/               # GRPO training plots
    └── sft_loss/                # SFT loss curves
```


## Episode Details

| Parameter | Value |
|---|---|
| Episode length | 90 minutes (5400 s) |
| Step duration | 30 seconds |
| Steps per episode | 180  |
| Initial battery | 90 / 100 |
| Initial attitude | sun |

## Actions

| # | Action | Parameters | Purpose |
|---|--------|------------|---------|
| 1 | `wait` | — | Do nothing for one step |
| 2 | `capture_image` | `target_id` | Capture imaging data for a target |
| 3 | `downlink_to_station` | — | Transmit compressed data to ground |
| 4 | `sun_point_for_charging` | — | Charge battery via solar panels |
| 5 | `compress_data` | — | Compress raw data on board |
| 6 | `abort_task` | — | Cancel current multi-step action |

## State Variables

- `current_time` — simulation clock (seconds)
- `attitude` — satellite pointing (`sun`, `gs`, `target_<id>`)
- `busy_status` — current task (`idle`, `slewing`, `capturing`, …)
- `remaining_action_steps` — steps left for multi-step action
- `battery_level` — 0–100
- `storage_used` — onboard storage in GB
- `raw_data_amount` / `compressed_data_amount` — data volumes
- `sunlit_status` / `ground_station_visible` — orbital booleans
- `pending_request_queue` — list of `TargetRequest` objects
- `current_selected_request_id`

## Building & Running

```bash
# Local (no Docker)
uv run server --host 0.0.0.0 --port 8000

# Docker
docker build -t satellite_scheduler-env:latest -f server/Dockerfile .
docker run -p 8000:8000 satellite_scheduler-env:latest

# Deploy to Hugging Face
openenv push
```
