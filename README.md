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
        SatelliteSchedulerAction(action_type=ActionType.SLEW_TO_TARGET, target_id=1)
    )
    print(f"Attitude: {result.observation.attitude}")
```

## Episode Details

| Parameter | Value |
|---|---|
| Episode length | 90 minutes (5 400 s) |
| Step duration | 30 seconds |
| Steps per episode | 180 |
| Initial battery | 90 / 100 |
| Initial attitude | sun |

## Actions

| # | Action | Parameters | Purpose |
|---|--------|------------|---------|
| 1 | `slew_to_target` | `target_id` | Repoint satellite to a target |
| 2 | `capture_image` | `target_id` | Capture imaging data for a target |
| 3 | `downlink_to_station` | — | Transmit compressed data to ground |
| 4 | `sun_point_for_charging` | — | Charge battery via solar panels |
| 5 | `compress_data` | — | Compress raw data on board |
| 6 | `abort_task` | — | Cancel current multi-step action |
| 7 | `wait` | — | Do nothing for one step |

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
