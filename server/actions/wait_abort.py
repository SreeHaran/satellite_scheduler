# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wait and abort action mixin for the Satellite Scheduler Environment."""

try:
    from ...models import (
        ABORT_BATTERY_COST,
        ABORT_PENALTY,
        ABORT_TIME_SEC,
        MAX_STEPS,
        WAIT_BATTERY_COST,
        WAIT_TIME_SEC,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        ABORT_BATTERY_COST,
        ABORT_PENALTY,
        ABORT_TIME_SEC,
        MAX_STEPS,
        WAIT_BATTERY_COST,
        WAIT_TIME_SEC,
    )


class WaitAbortMixin:
    """Mixin providing wait and abort action handling."""

    def _do_wait(self) -> float:
        self._current_time += WAIT_TIME_SEC
        self._drain_battery(WAIT_BATTERY_COST)
        self._total_stalling_steps += 1
        return -0.1 * (1.0 / MAX_STEPS)  # small penalty for stalling

    def _do_abort(self) -> float:
        self._current_time += ABORT_TIME_SEC
        self._drain_battery(ABORT_BATTERY_COST)
        self._busy_status = "idle"
        self._remaining_action_steps = 0
        self._current_action_type = None
        self._slew_destination = None
        self._slew_steps_left = (
            0  # TODO: check where the satellite is pointing where abort performed
        )
        self._total_aborts += 1
        return -0.1 * ABORT_PENALTY
