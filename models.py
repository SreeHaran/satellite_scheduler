"""
Data models for the Satellite Scheduler Environment.

Defines typed Action, Observation, and supporting schemas for a satellite
scheduling RL environment with imaging, compression, downlink, and charging.
"""

from enum import Enum
from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    WAIT = "wait"
    ABORT_TASK = "abort_task"
    SUN_POINT_FOR_CHARGING = "sun_point_for_charging"
    COMPRESS_DATA = "compress_data"
    DOWNLINK_TO_STATION = "downlink_to_station"
    CAPTURE_IMAGE = "capture_image"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImagingMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RequestStatus(str, Enum):
    PENDING = "pending"
    CAPTURED = "captured"
    COMPRESSED = "compressed"
    DOWNLINKED = "downlinked"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Supporting schemas
# ---------------------------------------------------------------------------


class TargetRequest(BaseModel):
    """An imaging request from the ground."""

    request_id: int = Field(..., description="Unique request identifier")
    arrival_time: int = Field(..., description="Time (seconds) when request arrives")
    priority: Priority = Field(..., description="Request priority: low | medium | high")
    imaging_mode: ImagingMode = Field(
        ..., description="Imaging resolution: low | medium | high"
    )
    deadline: int = Field(..., description="Deadline time (seconds) for this request")
    status: RequestStatus = Field(
        default=RequestStatus.PENDING, description="Current request status"
    )


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------


class SatelliteSchedulerAction(Action):
    """Action for the Satellite Scheduler environment."""

    action_type: ActionType = Field(..., description="The type of action to perform")
    target_id: Optional[int] = Field(
        default=None,
        description="Target request ID (required for capture_image to slew to target)",
    )


class SatelliteSchedulerObservation(Observation):
    """Observation returned after each step or reset."""

    current_time: int = Field(
        default=0, description="Current simulation time in seconds"
    )
    attitude: str = Field(
        default="sun", description="Current satellite pointing: sun | gs | target_<id>"
    )
    busy_status: str = Field(
        default="idle",
        description="Current task status: idle | capturing | downlinking | sun_pointing | compressing",
    )
    remaining_action_steps: int = Field(
        default=0, description="Steps left for the current multi-step action"
    )
    battery_level: float = Field(default=90.0, description="Battery level (0-100)")
    storage_used: float = Field(
        default=0.0, description="Total onboard storage used (GB)"
    )
    raw_data_amount: float = Field(default=0.0, description="Raw data on board (GB)")
    compressed_data_amount: float = Field(
        default=0.0, description="Compressed data on board (GB)"
    )
    sunlit_status: bool = Field(
        default=True, description="Whether the satellite is in sunlight"
    )
    ground_station_visible: bool = Field(
        default=False, description="Whether a ground station is in view"
    )
    pending_request_queue: List[TargetRequest] = Field(
        default_factory=list, description="Queue of imaging requests"
    )
    current_selected_request_id: Optional[int] = Field(
        default=None, description="Currently selected request ID"
    )


class SatelliteSchedulerState(State):
    """Extended environment state including post-episode statistics.

    Returned by the environment's ``state`` property. After episode end,
    ``episode_stats`` contains the accumulated grading metrics; during an
    episode it is an empty dict.
    """

    episode_stats: dict = Field(
        default_factory=dict,
        description="Accumulated episode statistics for grader evaluation",
    )


# ---------------------------------------------------------------------------
# Constants (shared between server and client for reference)
# ---------------------------------------------------------------------------

EPISODE_DURATION_SEC = 5400  # 90 minutes
STEP_DURATION_SEC = 30
MAX_STEPS = (
    EPISODE_DURATION_SEC // STEP_DURATION_SEC
)  # 180 steps per episode at 30s per step

BATTERY_MAX = 100.0
STORAGE_CAPACITY = 30.0  # GB

# Slew
SLEW_TIME_PER_STEP_SEC = 30
SLEW_BATTERY_PER_STEP = 1.5
SLEW_STEPS = {
    "target": {"target": 1, "sun": 3, "gs": 2},
    "sun": {"target": 3, "gs": 2, "sun": 0},
    "gs": {"target": 2, "sun": 2, "gs": 0},
}

# Capture
CAPTURE_STEP_TIME_SEC = 30
CAPTURE_STEP_BATTERY_COST = 3
CAPTURE_TIME_STEPS = {"low": 1, "medium": 2, "high": 3}
CAPTURE_BATTERY_COST = {"low": 1.0, "medium": 2.0, "high": 2.0}
CAPTURE_DATA_SIZE = {"low": 5.0, "medium": 10.0, "high": 15.0}

# Downlink
DOWNLINK_STEP_TIME_SEC = 30
DOWNLINK_STEP_BATTERY_COST = 4
DOWNLINK_STEP_DATA_GB = 4.0
DOWNLINK_MIN_BATTERY = 15.0

# Sun-pointing / charging
SUN_POINT_STEP_TIME_SEC = 30
SUN_CHARGE_PER_STEP = 10

# Compression
COMPRESS_STEP_TIME_SEC = 30
COMPRESS_STEP_BATTERY_COST = 2
COMPRESS_STEP_RAW_REDUCED = 3.0
COMPRESS_COMPRESSION_RATIO = 0.5
COMPRESS_MIN_BATTERY = 15.0

# Abort
ABORT_TIME_SEC = 30
ABORT_BATTERY_COST = 2
ABORT_PENALTY = 1.0

# Wait
WAIT_TIME_SEC = 30
WAIT_BATTERY_COST = 1

# Priority weights (for grading)
PRIORITY_WEIGHT = {"low": 1.0, "medium": 2.0, "high": 3.0}
