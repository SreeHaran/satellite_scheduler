# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Downlink action mixin — ground-station data transmission pipeline."""

import math

try:
    from ...models import (
        DOWNLINK_MIN_BATTERY,
        DOWNLINK_STEP_BATTERY_COST,
        DOWNLINK_STEP_DATA_GB,
        DOWNLINK_STEP_TIME_SEC,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )
    from ..orbit import _attitude_category
except ImportError:
    from models import (  # type: ignore[no-redef]
        DOWNLINK_MIN_BATTERY,
        DOWNLINK_STEP_BATTERY_COST,
        DOWNLINK_STEP_DATA_GB,
        DOWNLINK_STEP_TIME_SEC,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )
    from orbit import _attitude_category  # type: ignore[no-redef]


class DownlinkMixin:
    """Mixin providing ground-station downlink action handling."""

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

        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("gs", 0)
        downlink_data_steps = max(
            1, math.ceil(self._compressed_data_amount / DOWNLINK_STEP_DATA_GB)
        )
        total_steps = slew_steps + downlink_data_steps

        self._slew_steps_left = slew_steps
        self._remaining_action_steps = total_steps
        self._busy_status = "downlinking"
        self._current_action_type = "downlink"
        self._slew_destination = "gs"

        return self._continue_downlink()

    def _continue_downlink(self) -> float:
        self._current_time += DOWNLINK_STEP_TIME_SEC

        if self._slew_steps_left > 0:
            # Still slewing to GS — costs slew battery
            self._drain_battery(SLEW_BATTERY_PER_STEP)
            self._slew_steps_left -= 1
            self._remaining_action_steps -= 1
            if self._slew_steps_left <= 0:
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
            self._slew_steps_left = 0

        return reward
