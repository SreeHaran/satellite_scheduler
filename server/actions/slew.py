# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit slew action mixin for the Satellite Scheduler Environment."""

from typing import Optional

try:
    from ...models import (
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        SLEW_TIME_PER_STEP_SEC,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )
    from ..orbit import _attitude_category
except ImportError:
    from models import (  # type: ignore[no-redef]
        SLEW_BATTERY_PER_STEP,
        SLEW_STEPS,
        SLEW_TIME_PER_STEP_SEC,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )
    from orbit import _attitude_category  # type: ignore[no-redef]


class SlewMixin:
    """Mixin providing explicit slew-to-attitude action handling."""

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
        self._drain_battery(SLEW_BATTERY_PER_STEP)
        self._remaining_action_steps -= 1

        if self._remaining_action_steps <= 0:
            self._attitude = self._slew_destination or self._attitude
            self._busy_status = "idle"
            self._remaining_action_steps = 0
            self._current_action_type = None
            self._slew_destination = None

        return 0.0  # slew itself gives no reward
