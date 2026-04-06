# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step metrics collection mixin for the Satellite Scheduler Environment.

Accumulates statistics used by the graders.  Intended to be mixed into
SatelliteSchedulerEnvironment.
"""

try:
    from ..models import STORAGE_CAPACITY, ActionType
except ImportError:
    from models import STORAGE_CAPACITY, ActionType  # type: ignore[no-redef]


class MetricsMixin:
    """Mixin providing per-step statistics recording for grader evaluation."""

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
