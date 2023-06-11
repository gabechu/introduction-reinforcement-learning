def incremental_average_update(old_average: float, new_value: float, step: int) -> float:
    """Update the average value incrementally."""
    return old_average + 1 / step * (new_value - old_average)
