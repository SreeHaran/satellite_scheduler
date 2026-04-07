"""Orbital state management mixin for the Satellite Scheduler Environment.

Handles sunlit / ground-station visibility windows, request generation,
and request expiry.  Intended to be mixed into SatelliteSchedulerEnvironment.
"""

try:
    from ..models import (
        EPISODE_DURATION_SEC,
        STEP_DURATION_SEC,
        ImagingMode,
        Priority,
        RequestStatus,
        TargetRequest,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        EPISODE_DURATION_SEC,
        STEP_DURATION_SEC,
        ImagingMode,
        Priority,
        RequestStatus,
        TargetRequest,
    )


def _attitude_category(attitude: str) -> str:
    """Map an attitude string to a slew-table key: 'target', 'sun', or 'gs'."""
    if attitude.startswith("target"):
        return "target"
    if attitude == "gs":
        return "gs"
    return "sun"


class OrbitMixin:
    """Mixin providing orbital state updates and request lifecycle management."""

    def _update_orbital_state(self) -> None:
        """Update sunlit and ground-station visibility based on current_time.

        Alternating windows (step-based, 0-indexed):
          Sunlit:      0-30, 60-90, 130-150
          GS visible: 30-60, 90-120, 150-180
        """
        step = self._current_time // STEP_DURATION_SEC
        self._sunlit_status = (
            (0 <= step < 30) or (60 <= step < 90) or (130 <= step < 150)
        )
        self._ground_station_visible = (
            (30 <= step < 60) or (90 <= step < 120) or (150 <= step < 180)
        )

    def _expire_requests(self) -> None:
        """Mark pending requests past their deadline as expired, then prune the queue.

        Requests with status EXPIRED or CAPTURED are removed from the active
        queue; they remain accessible via the grader archive (_all_requests).
        """
        for req in self._pending_request_queue:
            if (
                req.status == RequestStatus.PENDING
                and self._current_time > req.deadline
            ):
                req.status = RequestStatus.EXPIRED

        self._pending_request_queue = [
            req
            for req in self._pending_request_queue
            if req.status == RequestStatus.PENDING
        ]

    def _generate_request(self, arrival: int) -> TargetRequest:
        """Generate a randomised imaging request for the episode."""
        priorities = list(Priority)
        modes = list(ImagingMode)
        deadline = arrival + self._rng.randint(
            EPISODE_DURATION_SEC // 4, EPISODE_DURATION_SEC - arrival
        )
        deadline = min(deadline, EPISODE_DURATION_SEC)
        return TargetRequest(
            request_id=self._rng.randint(1, 1000),
            arrival_time=arrival,
            priority=self._rng.choice(priorities),
            imaging_mode=self._rng.choice(modes),
            deadline=deadline,
            status=RequestStatus.PENDING,
        )
