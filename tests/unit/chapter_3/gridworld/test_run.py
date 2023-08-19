from unittest import TestCase

import numpy as np

from src.chapter_3.gridworld.run import Policy, calculate_state_value_function
from src.chapter_3.gridworld.state import State


class TestStateValueFunctionForRandomPolicy(TestCase):
    def setUp(self) -> None:
        self.state_value_matrix = np.array(
            [
                [3.3, 8.8, 4.4, 5.3, 1.5],
                [1.5, 3.0, 2.3, 1.9, 0.5],
                [0.1, 0.7, 0.7, 0.4, -0.4],
                [-1.0, 0.4, -0.4, -0.6, -1.2],
                [-1.9, -1.3, -1.2, -1.4, -2.0],
            ]
        )

    def test_state_A(self):
        actual = calculate_state_value_function(State(0, 1), self.state_value_matrix, 0.9, Policy.RANDOM_POLICY)
        assert actual == 8.83

    def test_state_B(self):
        actual = calculate_state_value_function(State(0, 3), self.state_value_matrix, 0.9, Policy.RANDOM_POLICY)
        assert actual == 5.36

    def test_state_00(self):
        actual = calculate_state_value_function(State(0, 0), self.state_value_matrix, 0.9, Policy.RANDOM_POLICY)
        assert actual == 3.3025

    def test_state_04(self):
        actual = calculate_state_value_function(State(0, 4), self.state_value_matrix, 0.9, Policy.RANDOM_POLICY)
        assert actual == 1.48

    def test_state_11(self):
        actual = calculate_state_value_function(State(1, 1), self.state_value_matrix, 0.9, Policy.RANDOM_POLICY)
        assert actual == 2.9925
