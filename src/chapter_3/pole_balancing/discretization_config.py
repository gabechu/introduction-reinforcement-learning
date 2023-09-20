from dataclasses import dataclass


@dataclass
class DiscretizationConfig:
    num_buckets_for_position: int
    num_buckets_for_angle: int
