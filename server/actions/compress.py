# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compression action mixin for the Satellite Scheduler Environment."""

import math

try:
    from ...models import (
        COMPRESS_COMPRESSION_RATIO,
        COMPRESS_MIN_BATTERY,
        COMPRESS_STEP_BATTERY_COST,
        COMPRESS_STEP_RAW_REDUCED,
        COMPRESS_STEP_TIME_SEC,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        COMPRESS_COMPRESSION_RATIO,
        COMPRESS_MIN_BATTERY,
        COMPRESS_STEP_BATTERY_COST,
        COMPRESS_STEP_RAW_REDUCED,
        COMPRESS_STEP_TIME_SEC,
        STEP_DURATION_SEC,
        WAIT_BATTERY_COST,
    )


class CompressMixin:
    """Mixin providing data compression action handling."""

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
        self._drain_battery(COMPRESS_STEP_BATTERY_COST)

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
