from src.chapter_2.ten_armed_bandits.utils import incremental_average_update


def test__incremental_average_update_for_one_step():
    actual = incremental_average_update(old_average=0, new_value=5, step=1)
    assert actual == 5


def test__incremental_average_update_for_two_steps():
    actual = incremental_average_update(old_average=1, new_value=3, step=2)
    assert actual == 2
