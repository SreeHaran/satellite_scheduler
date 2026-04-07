# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture action mixin — image acquisition pipeline."""

from typing import Optional

try:
    from ...models import (
        CAPTURE_BATTERY_COST,
        CAPTURE_DATA_SIZE,
        CAPTURE_STEP_BATTERY_COST,
        CAPTURE_STEP_TIME_SEC,
        CAPTURE_TIME_STEPS,
        PRIORITY_WEIGHT,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        WAIT_BATTERY_COST,
        RequestStatus,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        CAPTURE_BATTERY_COST,
        CAPTURE_DATA_SIZE,
        CAPTURE_STEP_BATTERY_COST,
        CAPTURE_STEP_TIME_SEC,
        CAPTURE_TIME_STEPS,
        PRIORITY_WEIGHT,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        STORAGE_CAPACITY,
        WAIT_BATTERY_COST,
        RequestStatus,
    )
from ..orbit import _attitude_category  # type: ignore[no-redef]


class CaptureMixin:
    """Mixin providing image capture action handling."""

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

        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("target", 0)

        self._current_selected_request_id = target_id
        self._slew_steps_left = slew_steps
        steps_needed = CAPTURE_TIME_STEPS[mode]
        self._remaining_action_steps = slew_steps + steps_needed
        self._busy_status = "capturing"
        self._current_action_type = "capture"

        return self._continue_capture()

    def _continue_capture(self) -> float:
        self._current_time += CAPTURE_STEP_TIME_SEC
        alive = self._drain_battery(CAPTURE_STEP_BATTERY_COST)
        self._remaining_action_steps -= 1

        # Phase 1: internal slew to target
        if self._slew_steps_left > 0:
            self._slew_steps_left -= 1

            if self._slew_steps_left == 0:
                self._attitude = f"target_{self._current_selected_request_id}"

            return 0.0 if alive else -1.0

        # Phase 2: actual capture
        if self._remaining_action_steps <= 0:
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
            self._slew_steps_left = 0

            return 0.5 * (
                PRIORITY_WEIGHT.get(req.priority.value, 1.0) / 3.0 if req else 0.0
            )

        return 0.0 if alive else -1.0
