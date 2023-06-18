from src.chapter_2.ten_armed_bandits.utils import weighted_incremental_average_update


def test__incremental_average_update_for_one_step():
    actual = weighted_incremental_average_update(old_average=0, new_value=5, weight=1)
    assert actual == 5


def test__incremental_average_update_for_two_steps():
    actual = weighted_incremental_average_update(old_average=1, new_value=3, weight=0.5)
    assert actual == 2
