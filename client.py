# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Satellite Scheduler Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    SatelliteSchedulerAction,
    SatelliteSchedulerObservation,
    SatelliteSchedulerState,
    TargetRequest,
)


class SatelliteSchedulerEnv(
    EnvClient[
        SatelliteSchedulerAction, SatelliteSchedulerObservation, SatelliteSchedulerState
    ]
):
    """
    Client for the Satellite Scheduler Environment.

    Example:
        >>> with SatelliteSchedulerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.battery_level)
        ...
        ...     action = SatelliteSchedulerAction(action_type="wait")
        ...     result = client.step(action)
        ...     print(result.observation.current_time)
    """

    def _step_payload(self, action: SatelliteSchedulerAction) -> Dict:
        payload: Dict = {"action_type": action.action_type.value}
        if action.target_id is not None:
            payload["target_id"] = action.target_id
        if action.destination is not None:
            payload["destination"] = action.destination
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SatelliteSchedulerObservation]:
        obs_data = payload.get("observation", {})

        request_queue_raw: List = obs_data.get("pending_request_queue", [])
        request_queue = [
            TargetRequest(**r) if isinstance(r, dict) else r for r in request_queue_raw
        ]

        observation = SatelliteSchedulerObservation(
            current_time=obs_data.get("current_time", 0),
            attitude=obs_data.get("attitude", "sun"),
            busy_status=obs_data.get("busy_status", "idle"),
            remaining_action_steps=obs_data.get("remaining_action_steps", 0),
            battery_level=obs_data.get("battery_level", 0.0),
            storage_used=obs_data.get("storage_used", 0.0),
            raw_data_amount=obs_data.get("raw_data_amount", 0.0),
            compressed_data_amount=obs_data.get("compressed_data_amount", 0.0),
            sunlit_status=obs_data.get("sunlit_status", True),
            ground_station_visible=obs_data.get("ground_station_visible", False),
            pending_request_queue=request_queue,
            current_selected_request_id=obs_data.get("current_selected_request_id"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SatelliteSchedulerState:
        return SatelliteSchedulerState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            episode_stats=payload.get("episode_stats", {}),
        )
