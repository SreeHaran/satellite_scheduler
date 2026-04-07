# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sun-pointing / charging action mixin for the Satellite Scheduler Environment."""

try:
    from ...models import (
        BATTERY_MAX,
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
            return -0.05

        src_cat = _attitude_category(self._attitude)
        slew_steps = SLEW_STEPS.get(src_cat, {}).get("sun", 0)

        self._slew_steps_left = slew_steps
        self._remaining_action_steps = (
            slew_steps  # charge is per-step; not pre-allocated
        )
        if slew_steps > 0:
            self._busy_status = "sun_pointing"
            self._current_action_type = "sun_point"
            self._slew_destination = "sun"

        return self._continue_sun_point()

    def _continue_sun_point(self) -> float:
        self._current_time += SUN_POINT_STEP_TIME_SEC

        if self._slew_steps_left > 0:
            # Slewing phase — satellite is locked in until pointing at sun
            self._drain_battery(SLEW_BATTERY_PER_STEP)
            self._slew_steps_left -= 1
            self._remaining_action_steps -= 1
            if self._slew_steps_left <= 0:
                self._attitude = "sun"
                self._busy_status = "idle"
                self._remaining_action_steps = 0
                self._current_action_type = None
                self._slew_destination = None
            return 0.0

        # Charging phase: one tick per action call; satellite stays idle so it can
        # choose any next action (including issuing sun_point_for_charging again).
        if not self._sunlit_status:
            self._drain_battery(WAIT_BATTERY_COST)
            return -0.2

        self._attitude = "sun"
        old_battery = self._battery_level
        self._battery_level = min(
            self._battery_level + SUN_CHARGE_PER_STEP, BATTERY_MAX
        )
        gained = self._battery_level - old_battery
        self._total_battery_gained += gained
        return 0.1
