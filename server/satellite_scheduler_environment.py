# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Satellite Scheduler Environment Implementation.

Simulates a low-Earth-orbit satellite that must image targets, compress data,
charge its battery, and downlink results within a 90-minute episode.
"""

import math
import random
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ABORT_BATTERY_COST,
        ABORT_PENALTY,
        ABORT_TIME_SEC,
        BATTERY_MAX,
        CAPTURE_BATTERY_COST,
        CAPTURE_DATA_SIZE,
        CAPTURE_STEP_BATTERY_COST,
        CAPTURE_STEP_TIME_SEC,
        CAPTURE_TIME_STEPS,
        COMPRESS_COMPRESSION_RATIO,
        COMPRESS_MIN_BATTERY,
        COMPRESS_STEP_BATTERY_COST,
        COMPRESS_STEP_RAW_REDUCED,
        COMPRESS_STEP_TIME_SEC,
        DOWNLINK_MIN_BATTERY,
        DOWNLINK_STEP_BATTERY_COST,
        DOWNLINK_STEP_DATA_GB,
        DOWNLINK_STEP_TIME_SEC,
        EPISODE_DURATION_SEC,
        MAX_STEPS,
        PRIORITY_WEIGHT,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        SLEW_TIME_PER_STEP_SEC,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        SUN_CHARGE_PER_STEP,
        SUN_POINT_STEP_TIME_SEC,
        WAIT_BATTERY_COST,
        WAIT_TIME_SEC,
        ActionType,
        ImagingMode,
        Priority,
        RequestStatus,
        SatelliteSchedulerAction,
        SatelliteSchedulerObservation,
        SatelliteSchedulerState,
        TargetRequest,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        ABORT_BATTERY_COST,
        ABORT_PENALTY,
        ABORT_TIME_SEC,
        BATTERY_MAX,
        CAPTURE_BATTERY_COST,
        CAPTURE_DATA_SIZE,
        CAPTURE_STEP_BATTERY_COST,
        CAPTURE_STEP_TIME_SEC,
        CAPTURE_TIME_STEPS,
        COMPRESS_COMPRESSION_RATIO,
        COMPRESS_MIN_BATTERY,
        COMPRESS_STEP_BATTERY_COST,
        COMPRESS_STEP_RAW_REDUCED,
        COMPRESS_STEP_TIME_SEC,
        DOWNLINK_MIN_BATTERY,
        DOWNLINK_STEP_BATTERY_COST,
        DOWNLINK_STEP_DATA_GB,
        DOWNLINK_STEP_TIME_SEC,
        EPISODE_DURATION_SEC,
        MAX_STEPS,
        PRIORITY_WEIGHT,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        SLEW_TIME_PER_STEP_SEC,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        SUN_CHARGE_PER_STEP,
        SUN_POINT_STEP_TIME_SEC,
        WAIT_BATTERY_COST,
        WAIT_TIME_SEC,
        ActionType,
        ImagingMode,
        Priority,
        RequestStatus,
        SatelliteSchedulerAction,
        SatelliteSchedulerObservation,
        SatelliteSchedulerState,
        TargetRequest,
    )


def _attitude_category(attitude: str) -> str:
    """Map an attitude string to a slew-table key: 'target', 'sun', or 'gs'."""
    if attitude.startswith("target"):
        return "target"
    if attitude == "gs":
        return "gs"
    return "sun"


class SatelliteSchedulerEnvironment(Environment):
    """
    Satellite Scheduler RL Environment.

    The agent controls a single LEO satellite over a 90-minute orbit.
    It must slew to targets, capture images, compress data, charge via
    sun-pointing, and downlink compressed data during ground-station passes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._state = SatelliteSchedulerState(episode_id=str(uuid4()), step_count=0)
        self._init_env_state()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _init_env_state(self) -> None:
        self._current_time: int = 0
        self._attitude: str = "sun"
        self._busy_status: str = "idle"
        self._remaining_action_steps: int = 0
        self._battery_level: float = 90.0
        self._storage_used: float = 0.0
        self._raw_data_amount: float = 0.0
        self._compressed_data_amount: float = 0.0
        self._sunlit_status: bool = True
        self._ground_station_visible: bool = False
        self._pending_request_queue: List[TargetRequest] = []
        self._current_selected_request_id: Optional[int] = None

        # Internal tracking for the current multi-step action
        self._current_action_type: Optional[str] = None
        self._slew_destination: Optional[str] = None  # where we are slewing to
        self._sun_point_slew_steps_left: int = (
            0  # slew steps remaining within sun-pointing
        )
        self._downlink_slew_steps_left: int = 0  # slew steps remaining within downlink

        # Reward tracking accumulators (for per-step reward)
        self._total_data_downlinked: float = 0.0
        self._total_data_generated: float = 0.0
        self._total_data_compressed: float = 0.0
        self._total_storage_freed_by_compression: float = 0.0
        self._total_battery_gained: float = 0.0
        self._total_invalid_actions: int = 0
        self._total_battery_violations: int = 0
        self._total_storage_violations: int = 0
        self._total_aborts: int = 0
        self._total_stalling_steps: int = 0
        self._step_number: int = 0

        # Grader statistics (accumulated each step)
        self._total_steps_recorded: int = 0
        self._storage_util_sum: float = 0.0  # sum of storage_used/capacity per step
        self._battery_low_steps: int = 0  # steps with battery < 10%
        self._storage_high_steps: int = 0  # steps with storage_used > 85 GB
        self._stalled_raw_steps: int = (
            0  # idle steps with raw_data > 0, not compressing
        )
        self._overflow_events: int = 0  # times storage exceeded capacity
        self._total_sunlit_steps: int = 0  # total steps in sunlight

    def _generate_requests(self) -> List[TargetRequest]:
        """Generate a randomised set of imaging requests for the episode."""
        priorities = list(Priority)
        modes = list(ImagingMode)
        num_requests = self._rng.randint(4, 8)
        requests: List[TargetRequest] = []
        for i in range(num_requests):
            arrival = self._rng.randint(0, EPISODE_DURATION_SEC // 2)
            deadline = arrival + self._rng.randint(
                EPISODE_DURATION_SEC // 4, EPISODE_DURATION_SEC - arrival
            )
            deadline = min(deadline, EPISODE_DURATION_SEC)
            requests.append(
                TargetRequest(
                    request_id=i + 1,
                    arrival_time=arrival,
                    priority=self._rng.choice(priorities),
                    imaging_mode=self._rng.choice(modes),
                    deadline=deadline,
                    status=RequestStatus.PENDING,
                )
            )
        return requests

    def _update_orbital_state(self) -> None:
        """Update sunlit and ground-station visibility based on current_time.

        Alternating windows (step-based, 0-indexed):
          Sunlit:      0-30, 60-90, 130-150
          GS visible: 30-60, 90-120, 150-180
        """
        step = self._current_time // STEP_DURATION_SEC
        self._sunlit_status = (
            (0 <= step < 30) or (60 <= step < 90) or (130 <= step < 150)
        )
        self._ground_station_visible = (
            (30 <= step < 60) or (90 <= step < 120) or (150 <= step < 180)
        )

    def _expire_requests(self) -> None:
        """Mark pending requests past their deadline as expired."""
        for req in self._pending_request_queue:
            if (
                req.status == RequestStatus.PENDING
                and self._current_time > req.deadline
            ):
                req.status = RequestStatus.EXPIRED

    def _get_request(self, request_id: int) -> Optional[TargetRequest]:
        for req in self._pending_request_queue:
            if req.request_id == request_id:
                return req
        return None

    def _build_observation(
        self, reward: float, done: bool
    ) -> SatelliteSchedulerObservation:
        return SatelliteSchedulerObservation(
            current_time=self._current_time,
            attitude=self._attitude,
            busy_status=self._busy_status,
            remaining_action_steps=self._remaining_action_steps,
            battery_level=round(self._battery_level, 2),
            storage_used=round(self._storage_used, 2),
            raw_data_amount=round(self._raw_data_amount, 2),
            compressed_data_amount=round(self._compressed_data_amount, 2),
            sunlit_status=self._sunlit_status,
            ground_station_visible=self._ground_station_visible,
            pending_request_queue=self._pending_request_queue,
            current_selected_request_id=self._current_selected_request_id,
            done=done,
            reward=round(reward, 4),
        )

    # ------------------------------------------------------------------
    # Battery helper
    # ------------------------------------------------------------------

    def _drain_battery(self, amount: float) -> bool:
        """Drain battery. Returns False if battery would go to zero (episode ends)."""
        self._battery_level -= amount
        if self._battery_level <= 0:
            self._battery_level = 0.0
            return False
        return True

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> SatelliteSchedulerObservation:
        self._state = SatelliteSchedulerState(episode_id=str(uuid4()), step_count=0)
        self._init_env_state()
        self._pending_request_queue = self._generate_requests()
        self._update_orbital_state()
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SatelliteSchedulerAction) -> SatelliteSchedulerObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._step_number += 1

        # Record pre-action state for grader metrics
        self._record_step_metrics(action.action_type)

        reward = 0.0
        done = False

        # ---- If satellite is busy, only allow continue (same action) or abort ----
        if self._busy_status != "idle":
            reward, done = self._handle_busy_step(action)
        else:
            reward, done = self._handle_idle_step(action)

        # Overflow detection (post-action)
        if self._storage_used > STORAGE_CAPACITY:
            self._overflow_events += 1

        # Advance orbital state
        self._update_orbital_state()
        self._expire_requests()

        # Check episode end
        if self._current_time >= EPISODE_DURATION_SEC:
            done = True

        # Battery death - terminate the episode
        if self._battery_level <= 0:
            done = True
            reward = -10.0

        return self._build_observation(reward=reward, done=done)

    @property
    def state(self) -> SatelliteSchedulerState:
        """Return current session state including accumulated episode statistics."""
        total_steps = max(1, self._total_steps_recorded)
        self._state.episode_stats = {
            "total_steps": total_steps,
            "total_data_downlinked": self._total_data_downlinked,
            "total_data_generated": self._total_data_generated,
            "total_data_compressed": self._total_data_compressed,
            "total_battery_gained": self._total_battery_gained,
            "total_sunlit_steps": self._total_sunlit_steps,
            "storage_util_sum": self._storage_util_sum,
            "battery_low_steps": self._battery_low_steps,
            "storage_high_steps": self._storage_high_steps,
            "stalled_raw_steps": self._stalled_raw_steps,
            "overflow_events": self._overflow_events,
            "requests": [r.model_dump() for r in self._pending_request_queue],
        }
        return self._state

    # ------------------------------------------------------------------
    # Grader metric collection
    # ------------------------------------------------------------------

    def _record_step_metrics(self, action_type: ActionType) -> None:
        """Record per-step statistics before applying the action."""
        self._total_steps_recorded += 1

        # Storage utilization snapshot
        self._storage_util_sum += self._storage_used / STORAGE_CAPACITY

        # Battery critically low
        if self._battery_level < 10.0:
            self._battery_low_steps += 1

        # Storage critically full (> 85 GB)
        if self._storage_used > 85.0:
            self._storage_high_steps += 1

        # Sunlit window
        if self._sunlit_status:
            self._total_sunlit_steps += 1

        # Stalled: had raw data, was idle, but chose something other than compress
        if (
            self._raw_data_amount > 0
            and self._busy_status == "idle"
            and action_type != ActionType.COMPRESS_DATA
        ):
            self._stalled_raw_steps += 1

    # ------------------------------------------------------------------
    # Busy-step handling (continue or abort an in-progress action)
    # ------------------------------------------------------------------

    def _handle_busy_step(self, action: SatelliteSchedulerAction) -> tuple[float, bool]:
        """Handle a step when the satellite is already busy with a multi-step task."""
        reward = 0.0
        done = False

        # Abort
        if action.action_type == ActionType.ABORT_TASK:
            return self._do_abort(), done

        # Continue current task (any action other than abort while busy = continue)
        if self._busy_status == "slewing":
            reward = self._continue_slew()
        elif self._busy_status == "capturing":
            reward = self._continue_capture()
        elif self._busy_status == "downlinking":
            reward = self._continue_downlink()
        elif self._busy_status == "sun_pointing":
            reward = self._continue_sun_point()
        elif self._busy_status == "compressing":
            reward = self._continue_compress()

        return reward, done

    # ------------------------------------------------------------------
    # Idle-step handling (start a new action)
    # ------------------------------------------------------------------

    def _handle_idle_step(self, action: SatelliteSchedulerAction) -> tuple[float, bool]:
        reward = 0.0
        done = False

        at = action.action_type

        if at == ActionType.WAIT:
            reward = self._do_wait()
        elif at == ActionType.ABORT_TASK:
            # Abort when idle -> invalid
            self._total_invalid_actions += 1
            reward = -0.4
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
        elif at == ActionType.SUN_POINT_FOR_CHARGING:
            reward = self._do_sun_point_start()
        elif at == ActionType.COMPRESS_DATA:
            reward = self._do_compress_start()
        elif at == ActionType.DOWNLINK_TO_STATION:
            reward = self._do_downlink_start()
        elif at == ActionType.SLEW_TO_TARGET:
            reward = self._do_slew_start(action.target_id, action.destination)
        elif at == ActionType.CAPTURE_IMAGE:
            reward = self._do_capture_start(action.target_id)
        else:
            self._total_invalid_actions += 1
            reward = -0.4
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)

        return reward, done

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    # ---- WAIT ----
    def _do_wait(self) -> float:
        self._current_time += WAIT_TIME_SEC
        self._drain_battery(WAIT_BATTERY_COST)
        self._total_stalling_steps += 1
        return -0.1 * (1.0 / MAX_STEPS)  # small penalty for stalling

    # ---- ABORT ----
    def _do_abort(self) -> float:
        self._current_time += ABORT_TIME_SEC
        self._drain_battery(ABORT_BATTERY_COST)
        self._busy_status = "idle"
        self._remaining_action_steps = 0
        self._current_action_type = None
        self._slew_destination = None
        self._total_aborts += 1
        return -0.1 * ABORT_PENALTY

    # ---- SLEW ----
    def _do_slew_start(
        self, target_id: Optional[int], destination: Optional[str] = None
    ) -> float:
        # Determine slew destination
        if destination in ("sun", "gs"):
            dest = destination
            dst_cat = destination
        elif target_id is not None:
            req = self._get_request(target_id)
            if req is None:
                self._total_invalid_actions += 1
                self._current_time += STEP_DURATION_SEC
                self._drain_battery(WAIT_BATTERY_COST)
                return -0.4
            dest = f"target_{target_id}"
            dst_cat = "target"
            self._current_selected_request_id = target_id
        else:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        src_cat = _attitude_category(self._attitude)
        steps_needed = SLEW_STEPS.get(src_cat, {}).get(dst_cat, 2)

        if steps_needed == 0:
            # Already at this category — still costs 1 step for target-to-target
            if dst_cat == "target":
                steps_needed = 1
            else:
                # Already pointing here, no-op
                self._attitude = dest
                self._current_time += STEP_DURATION_SEC
                self._drain_battery(WAIT_BATTERY_COST)
                return 0.0

        self._slew_destination = dest
        self._remaining_action_steps = steps_needed
        self._busy_status = "slewing"
        self._current_action_type = "slew"

        return self._continue_slew()

    def _continue_slew(self) -> float:
        self._current_time += SLEW_TIME_PER_STEP_SEC
        alive = self._drain_battery(SLEW_BATTERY_PER_STEP)
        self._remaining_action_steps -= 1

        if self._remaining_action_steps <= 0:
            self._attitude = self._slew_destination or self._attitude
            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None
            self._slew_destination = None

        return 0.0  # slew itself gives no reward

    # ---- CAPTURE ----
    def _do_capture_start(self, target_id: Optional[int]) -> float:
        if target_id is None:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        req = self._get_request(target_id)
        if req is None:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        expected_attitude = f"target_{target_id}"
        if self._attitude != expected_attitude:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        if req.status != RequestStatus.PENDING:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        mode = req.imaging_mode.value
        total_battery_cost = CAPTURE_BATTERY_COST[mode]
        data_size = CAPTURE_DATA_SIZE[mode]

        if self._battery_level < total_battery_cost:
            self._total_battery_violations += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.25

        if self._storage_used + data_size > STORAGE_CAPACITY:
            self._total_storage_violations += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.15

        self._current_selected_request_id = target_id
        steps_needed = CAPTURE_TIME_STEPS[mode]
        self._remaining_action_steps = steps_needed
        self._busy_status = "capturing"
        self._current_action_type = "capture"

        return self._continue_capture()

    def _continue_capture(self) -> float:
        self._current_time += CAPTURE_STEP_TIME_SEC
        alive = self._drain_battery(CAPTURE_STEP_BATTERY_COST)
        self._remaining_action_steps -= 1

        if self._remaining_action_steps <= 0:
            # Capture complete
            req = self._get_request(self._current_selected_request_id)  # type: ignore[arg-type]
            if req is not None:
                mode = req.imaging_mode.value
                data_size = CAPTURE_DATA_SIZE[mode]
                self._raw_data_amount += data_size
                self._storage_used += data_size
                self._total_data_generated += data_size
                req.status = RequestStatus.CAPTURED

            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None
            # Reward for image captured (normalized)
            return 0.3 * (
                PRIORITY_WEIGHT.get(req.priority.value, 1.0) / 3.0 if req else 0.0
            )

        return 0.0

    # ---- DOWNLINK ----
    def _do_downlink_start(self) -> float:
        if not self._ground_station_visible:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        if self._compressed_data_amount <= 0:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        if self._battery_level < DOWNLINK_MIN_BATTERY:
            self._total_battery_violations += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.25

        # If not already pointing at GS, need to slew first
        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("gs", 0)
        downlink_data_steps = max(
            1, math.ceil(self._compressed_data_amount / DOWNLINK_STEP_DATA_GB)
        )
        total_steps = slew_steps + downlink_data_steps

        self._downlink_slew_steps_left = slew_steps
        self._remaining_action_steps = total_steps
        self._busy_status = "downlinking"
        self._current_action_type = "downlink"
        self._slew_destination = "gs"

        return self._continue_downlink()

    def _continue_downlink(self) -> float:
        self._current_time += DOWNLINK_STEP_TIME_SEC

        if self._downlink_slew_steps_left > 0:
            # Still slewing to GS — costs slew battery
            self._drain_battery(SLEW_BATTERY_PER_STEP)
            self._downlink_slew_steps_left -= 1
            self._remaining_action_steps -= 1
            if self._downlink_slew_steps_left <= 0:
                self._attitude = "gs"
            return 0.0

        # Actual downlink phase
        self._attitude = "gs"
        self._drain_battery(DOWNLINK_STEP_BATTERY_COST)

        data_sent = min(DOWNLINK_STEP_DATA_GB, self._compressed_data_amount)
        self._compressed_data_amount -= data_sent
        self._storage_used -= data_sent
        self._total_data_downlinked += data_sent
        self._remaining_action_steps -= 1

        reward = 0.5 * (data_sent / DOWNLINK_STEP_DATA_GB)  # normalized per-step

        if self._remaining_action_steps <= 0 or self._compressed_data_amount <= 0:
            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None
            self._slew_destination = None
            self._downlink_slew_steps_left = 0

        return reward

    # ---- SUN POINT ----
    def _do_sun_point_start(self) -> float:
        if not self._sunlit_status:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        if self._battery_level >= BATTERY_MAX:
            # Already full, just waste a step
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            self._total_stalling_steps += 1
            return -0.1 * (1.0 / MAX_STEPS)

        # If not already pointing at sun, need to slew first
        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("sun", 0)
        charge_steps = max(
            1, math.ceil((BATTERY_MAX - self._battery_level) / SUN_CHARGE_PER_STEP)
        )
        total_steps = slew_steps + charge_steps

        self._sun_point_slew_steps_left = slew_steps
        self._remaining_action_steps = total_steps
        self._busy_status = "sun_pointing"
        self._current_action_type = "sun_point"
        self._slew_destination = "sun"

        return self._continue_sun_point()

    def _continue_sun_point(self) -> float:
        self._current_time += SUN_POINT_STEP_TIME_SEC

        if self._sun_point_slew_steps_left > 0:
            # Still slewing to sun — costs battery
            self._drain_battery(SLEW_BATTERY_PER_STEP)
            self._sun_point_slew_steps_left -= 1
            self._remaining_action_steps -= 1
            if self._sun_point_slew_steps_left <= 0:
                self._attitude = "sun"
            reward = 0.0
        else:
            # Charging phase
            self._attitude = "sun"
            old_battery = self._battery_level
            self._battery_level = min(
                self._battery_level + SUN_CHARGE_PER_STEP, BATTERY_MAX
            )
            gained = self._battery_level - old_battery
            self._total_battery_gained += gained
            self._remaining_action_steps -= 1
            reward = 0.1 * (gained / BATTERY_MAX)

        if self._remaining_action_steps <= 0 or self._battery_level >= BATTERY_MAX:
            self._attitude = "sun"
            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None
            self._slew_destination = None
            self._sun_point_slew_steps_left = 0

        return reward

    # ---- COMPRESS ----
    def _do_compress_start(self) -> float:
        if self._raw_data_amount <= 0:
            self._total_invalid_actions += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.4

        if self._battery_level < COMPRESS_MIN_BATTERY:
            self._total_battery_violations += 1
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.25

        steps_needed = max(
            1, math.ceil(self._raw_data_amount / COMPRESS_STEP_RAW_REDUCED)
        )
        self._remaining_action_steps = steps_needed
        self._busy_status = "compressing"
        self._current_action_type = "compress"

        return self._continue_compress()

    def _continue_compress(self) -> float:
        self._current_time += COMPRESS_STEP_TIME_SEC
        alive = self._drain_battery(COMPRESS_STEP_BATTERY_COST)

        raw_reduced = min(COMPRESS_STEP_RAW_REDUCED, self._raw_data_amount)
        compressed_gained = raw_reduced * COMPRESS_COMPRESSION_RATIO
        storage_freed = raw_reduced - compressed_gained

        self._raw_data_amount -= raw_reduced
        self._compressed_data_amount += compressed_gained
        self._storage_used -= storage_freed
        self._total_data_compressed += compressed_gained
        self._total_storage_freed_by_compression += storage_freed
        self._remaining_action_steps -= 1

        reward = 0.1 * (storage_freed / COMPRESS_STEP_RAW_REDUCED)  # normalized

        if self._remaining_action_steps <= 0 or self._raw_data_amount <= 0:
            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None

        return reward
