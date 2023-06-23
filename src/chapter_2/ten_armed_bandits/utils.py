def weighted_incremental_average_update(old_average: float, new_value: float, weight: float) -> float:
    """Update the average value incrementally."""
    return old_average + weight * (new_value - old_average)
