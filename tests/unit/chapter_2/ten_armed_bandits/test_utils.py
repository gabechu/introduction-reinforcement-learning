from src.chapter_2.ten_armed_bandits.utils import weighted_incremental_average_update


def test_weighted_incremental_average_update_with_weight_of_one():
    actual = weighted_incremental_average_update(old_average=1, new_value=5, weight=1)
    assert actual == 5


def test_weighted_incremental_average_update_with_weight_less_than_one():
    actual = weighted_incremental_average_update(old_average=1, new_value=5, weight=0.5)
    assert actual == 3


def test_weighted_incremental_average_update_with_weight_greater_than_one():
    actual = weighted_incremental_average_update(old_average=1, new_value=5, weight=2)
    assert actual == 9


def test_weighted_incremental_average_update_for_sample_average():
    actual = []
    targets = [1, 2, 3, 4, 5]
    avg = 0
    for i, value in enumerate(targets):
        new_avg = weighted_incremental_average_update(old_average=avg, new_value=value, weight=1 / (i + 1))
        actual.append(new_avg)
        avg = new_avg

    assert actual == [1.0, 1.5, 2.0, 2.5, 3.0]
