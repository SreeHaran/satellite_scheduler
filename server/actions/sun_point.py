# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sun-pointing / charging action mixin for the Satellite Scheduler Environment."""

import math

try:
    from ...models import (
        BATTERY_MAX,
        MAX_STEPS,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        SUN_CHARGE_PER_STEP,
        SUN_POINT_STEP_TIME_SEC,
        WAIT_BATTERY_COST,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        BATTERY_MAX,
        MAX_STEPS,
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        STEP_DURATION_SEC,
        SUN_CHARGE_PER_STEP,
        SUN_POINT_STEP_TIME_SEC,
        WAIT_BATTERY_COST,
    )
from ..orbit import _attitude_category


class SunPointMixin:
    """Mixin providing sun-pointing and battery charging action handling."""

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

        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("sun", 0)
        charge_steps = max(
            1, math.ceil((BATTERY_MAX - self._battery_level) / SUN_CHARGE_PER_STEP)
        )
        total_steps = slew_steps + charge_steps

        self._slew_steps_left = slew_steps
        self._remaining_action_steps = total_steps
        self._busy_status = "sun_pointing"
        self._current_action_type = "sun_point"
        self._slew_destination = "sun"

        return self._continue_sun_point()

    def _continue_sun_point(self) -> float:
        self._current_time += SUN_POINT_STEP_TIME_SEC

        if self._slew_steps_left > 0:
            # Still slewing to sun — costs battery
            self._drain_battery(SLEW_BATTERY_PER_STEP)
            self._slew_steps_left -= 1
            self._remaining_action_steps -= 1
            if self._slew_steps_left <= 0:
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
            self._slew_steps_left = 0

        return reward
