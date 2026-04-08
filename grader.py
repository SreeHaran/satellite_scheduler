"""
Graders for the Satellite Scheduler Environment.

Three difficulty levels defined by the spec:
  - Easy   : priority-weighted mission completion ratio
  - Medium : storage-flow efficiency
  - Hard   : closed-loop mission planning (all subsystems)
"""

from typing import Any, Dict

try:
    from .models import PRIORITY_WEIGHT, SUN_CHARGE_PER_STEP
except ImportError:
    from models import PRIORITY_WEIGHT, SUN_CHARGE_PER_STEP  # type: ignore[no-redef]

# Statuses considered "completed" for priority scoring
_COMPLETED_STATUSES = {"captured", "compressed", "downlinked"}


def _clamp(value: float) -> float:
    return max(0.0001, min(0.9999, value))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0 else default


# ---------------------------------------------------------------------------
# Grader 1 – Easy: Priority-aware mission planning
# ---------------------------------------------------------------------------


def grade_easy(episode_stats: Dict[str, Any]) -> float:
    """
    Score = sum(priority_weight for completed requests)
             / sum(priority_weight for all requests)

    "Completed" means the image was captured (status not pending or expired).
    Returns a value in [0, 1].
    """
    requests = episode_stats.get("requests", [])
    if not requests:
        return 0.0

    w_total = sum(PRIORITY_WEIGHT.get(r["priority"], 1.0) for r in requests)
    w_done = sum(
        PRIORITY_WEIGHT.get(r["priority"], 1.0)
        for r in requests
        if r["status"] in _COMPLETED_STATUSES
    )

    return round(_clamp(_safe_div(w_done, w_total)), 4)


# ---------------------------------------------------------------------------
# Grader 2 – Medium: Storage-flow efficiency
# ---------------------------------------------------------------------------


def grade_medium(episode_stats: Dict[str, Any]) -> float:
    """
    score = 0.50 * (data_downlinked / data_compressed)
          + 0.30 * (data_compressed / data_generated)
          + 0.20 * (1 - avg_storage_utilization)
          - 0.15 * stalled_raw_data_fraction
          - 0.10 * storage_full_steps_fraction

    Positive weights sum to 1.0, so best case = 1.0.
    Returns a value in [0, 1].
    """
    total_steps = episode_stats["total_steps"]

    data_downlinked = episode_stats["total_data_downlinked"]
    data_compressed = episode_stats["total_data_compressed"]
    data_generated = episode_stats["total_data_generated"]

    avg_storage_util = _safe_div(episode_stats["storage_util_sum"], total_steps)
    stalled_frac = _safe_div(episode_stats["stalled_raw_steps"], total_steps)
    full_frac = _safe_div(episode_stats["storage_high_steps"], total_steps)

    score = (
        0.50 * _safe_div(data_downlinked, data_compressed)
        + 0.30 * _safe_div(data_compressed, data_generated)
        + 0.20 * (1.0 - avg_storage_util)
        - 0.15 * stalled_frac
        - 0.10 * full_frac
    )

    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Grader 3 – Hard: Closed-loop mission planning
# ---------------------------------------------------------------------------


def grade_hard(episode_stats: Dict[str, Any]) -> float:
    """
    downlinked_data_fraction = total_data_downlinked / total_data_generated
    charging_efficiency      = min(1, useful_charge / max_possible_charge)
    storage_efficiency       = 1 - (overflow_events + idle_full_steps) / total_steps

    score = 0.50 * (W_done / W_total)
          + 0.30 * downlinked_data_fraction
          + 0.10 * charging_efficiency
          + 0.10 * storage_efficiency
          - 0.15 * missed_deadlines_fraction
          - 0.20 * battery_low_fraction
          - 0.20 * storage_high_fraction

    Returns a value roughly in [0, 1] (can be negative for poor episodes).
    """
    total_steps = episode_stats["total_steps"]
    requests = episode_stats.get("requests", [])

    # W_done / W_total – priority-weighted captured requests
    w_total = sum(PRIORITY_WEIGHT.get(r["priority"], 1.0) for r in requests)
    w_done = sum(
        PRIORITY_WEIGHT.get(r["priority"], 1.0)
        for r in requests
        if r["status"] in _COMPLETED_STATUSES
    )
    w_ratio = _safe_div(w_done, w_total)

    # Downlinked data fraction
    downlinked_frac = _safe_div(
        episode_stats["total_data_downlinked"],
        episode_stats["total_data_generated"],
    )

    # Charging efficiency: useful charge / max possible charge in sunlit windows
    max_possible_charge = episode_stats["total_sunlit_steps"] * SUN_CHARGE_PER_STEP
    charging_efficiency = min(
        1.0, _safe_div(episode_stats["total_battery_gained"], max_possible_charge)
    )

    # Storage efficiency: penalise overflow events and steps at full storage
    overflow_events = episode_stats["overflow_events"]
    idle_full_steps = episode_stats["storage_high_steps"]
    storage_efficiency = 1.0 - _safe_div(overflow_events + idle_full_steps, total_steps)

    # Missed deadlines fraction
    num_requests = len(requests)
    missed = sum(1 for r in requests if r["status"] == "expired")
    missed_frac = _safe_div(missed, num_requests)

    # Battery and storage violation fractions
    battery_low_frac = _safe_div(episode_stats["battery_low_steps"], total_steps)
    storage_high_frac = _safe_div(episode_stats["storage_high_steps"], total_steps)

    score = (
        0.50 * w_ratio
        + 0.30 * downlinked_frac
        + 0.10 * charging_efficiency
        + 0.10 * storage_efficiency
        - 0.15 * missed_frac
        - 0.20 * battery_low_frac
        - 0.20 * storage_high_frac
    )

    return round(_clamp(score), 4)


# ---------------------------------------------------------------------------
# Convenience: grade all three at once
# ---------------------------------------------------------------------------


def grade_all(episode_stats: Dict[str, Any]) -> Dict[str, float]:
    """Return scores for all three graders as a dict."""
    return {
        "easy": grade_easy(episode_stats),
        "medium": grade_medium(episode_stats),
        "hard": grade_hard(episode_stats),
    }
