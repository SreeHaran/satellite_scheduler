# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Satellite Scheduler Environment Implementation.

Simulates a low-Earth-orbit satellite that must image targets, compress data,
charge its battery, and downlink results within a 90-minute episode.

Action handlers, orbital logic, and metrics collection are split into
focused mixin modules:
    - orbit.py             — orbital state & request lifecycle
    - metrics.py           — per-step grader statistics
    - actions/wait_abort.py — wait/abort controls
    - actions/capture.py   — image capture pipeline
    - actions/downlink.py  — ground-station downlink
    - actions/sun_point.py — sun-pointing / charging
    - actions/compress.py  — data compression
"""

import random
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        EPISODE_DURATION_SEC,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        WAIT_BATTERY_COST,
        ActionType,
        SatelliteSchedulerAction,
        SatelliteSchedulerObservation,
        SatelliteSchedulerState,
        TargetRequest,
    )
    from .actions.capture import CaptureMixin
    from .actions.compress import CompressMixin
    from .actions.downlink import DownlinkMixin
    from .actions.sun_point import SunPointMixin
    from .actions.wait_abort import WaitAbortMixin
    from .metrics import MetricsMixin
    from .orbit import OrbitMixin
except ImportError:
    from models import (  # type: ignore[no-redef]
        EPISODE_DURATION_SEC,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        WAIT_BATTERY_COST,
        ActionType,
        SatelliteSchedulerAction,
        SatelliteSchedulerObservation,
        SatelliteSchedulerState,
        TargetRequest,
    )
    from server.actions.capture import CaptureMixin  # type: ignore[no-redef]
    from server.actions.compress import CompressMixin  # type: ignore[no-redef]
    from server.actions.downlink import DownlinkMixin  # type: ignore[no-redef]
    from server.actions.sun_point import SunPointMixin  # type: ignore[no-redef]
    from server.actions.wait_abort import WaitAbortMixin  # type: ignore[no-redef]
    from server.metrics import MetricsMixin  # type: ignore[no-redef]
    from server.orbit import OrbitMixin  # type: ignore[no-redef]


class SatelliteSchedulerEnvironment(
    OrbitMixin,
    MetricsMixin,
    WaitAbortMixin,
    CaptureMixin,
    DownlinkMixin,
    SunPointMixin,
    CompressMixin,
    Environment,
):
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
    # Internal state initialisation
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
        self._all_requests: List[TargetRequest] = []  # archive for grader tracking
        self._current_selected_request_id: Optional[int] = None

        # Internal tracking for the current multi-step action
        self._current_action_type: Optional[str] = None
        self._slew_destination: Optional[str] = None  # where we are slewing to

        # Single shared slew-phase counter; only one action can be active at a time.
        self._slew_steps_left: int = 0

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
        self._storage_high_steps: int = 0  # steps with storage_used > 85% of capacity
        self._stalled_raw_steps: int = (
            0  # idle steps with raw_data > 0, not compressing
        )
        self._overflow_events: int = 0  # times storage exceeded capacity
        self._total_sunlit_steps: int = 0  # total steps in sunlight

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _drain_battery(self, amount: float) -> bool:
        """Drain battery. Returns False if battery would go to zero (episode ends)."""
        self._battery_level -= amount
        if self._battery_level <= 1:
            self._battery_level = 0.0
            return False
        return True

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> SatelliteSchedulerObservation:
        self._state = SatelliteSchedulerState(episode_id=str(uuid4()), step_count=0)
        self._init_env_state()
        self._pending_request_queue = []
        self._all_requests = []
        self._update_orbital_state()
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SatelliteSchedulerAction) -> SatelliteSchedulerObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._step_number += 1

        # Randomly generate target requests every 5 steps
        if self._step_number % 5 == 0:
            last_5_minutes = EPISODE_DURATION_SEC - 300
            if (
                len(self._pending_request_queue) < 7  # no more than 7 requests
                and self._current_time < last_5_minutes  # less than 5 minutes left
                and self._rng.random() < 0.9  # 90% chance to generate a new request
            ):
                new_request = self._generate_request(self._current_time)
                self._pending_request_queue.append(new_request)
                self._all_requests.append(new_request)

        # Record pre-action state for grader metrics
        self._record_step_metrics(action.action_type)

        reward = 0.0
        done = False

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

        if self._current_time >= EPISODE_DURATION_SEC:
            done = True

        # Battery death — terminate the episode
        if self._battery_level <= 1:
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
            "requests": [r.model_dump() for r in self._all_requests],
        }
        return self._state

    # ------------------------------------------------------------------
    # Busy-step dispatch (continue or abort an in-progress action)
    # ------------------------------------------------------------------

    def _handle_busy_step(self, action: SatelliteSchedulerAction) -> tuple[float, bool]:
        """Handle a step when the satellite is busy with a multi-step task."""
        if action.action_type == ActionType.ABORT_TASK:
            return self._do_abort(), False

        # Any action other than abort while busy = continue current task
        if self._busy_status == "capturing":
            reward = self._continue_capture()
        elif self._busy_status == "downlinking":
            reward = self._continue_downlink()
        elif self._busy_status == "sun_pointing":
            reward = self._continue_sun_point()
        elif self._busy_status == "compressing":
            reward = self._continue_compress()
        else:
            reward = 0.0

        return reward, False

    # ------------------------------------------------------------------
    # Idle-step dispatch (start a new action)
    # ------------------------------------------------------------------

    def _handle_idle_step(self, action: SatelliteSchedulerAction) -> tuple[float, bool]:
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
        elif at == ActionType.CAPTURE_IMAGE:
            reward = self._do_capture_start(action.target_id)
        else:
            self._total_invalid_actions += 1
            reward = -0.4
            self._current_time += STEP_DURATION_SEC
            self._drain_battery(WAIT_BATTERY_COST)

        return reward, False
