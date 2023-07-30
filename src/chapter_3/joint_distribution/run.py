# | Temperature | Weather | P(T, W) |
# |-------------|---------|---------|
# | hot         | sun     | 0.70    |
# | hot         | rain    | 0.10    |
# | cold        | sun     | 0.05    |
# | cold        | rain    | 0.15    |

from typing import Dict, Tuple
from enum import Enum


class Temperature(str, Enum):
    HOT = "hot"
    COLD = "cold"


class Weather(str, Enum):
    SUN = "sun"
    RAIN = "rain"


def get_temperature_distribution(
    joint_distribution: Dict[Tuple[Temperature, Weather], float]
) -> Dict[Temperature, float]:
    return {
        temperature: sum(
            joint_distribution[temperature, weather] for weather in Weather
        )
        for temperature in Temperature
    }


def get_weather_distribution(
    joint_distribution: Dict[Tuple[Temperature, Weather], float]
) -> Dict[Weather, float]:
    return {
        weather: sum(
            joint_distribution[temperature, weather] for temperature in Temperature
        )
        for weather in Weather
    }


def get_conditional_distribution(
    joint_distribution: Dict[Tuple[Temperature, Weather], float]
) -> Dict[Tuple, float]:
    temperature_distribution = get_temperature_distribution(joint_distribution)
    weather_distribution = get_weather_distribution(joint_distribution)

    conditional_distribution = {}
    for temperature in Temperature:
        for weather in Weather:
            conditional_distribution[(temperature, weather)] = (
                joint_distribution[temperature, weather] / weather_distribution[weather]
            )
            conditional_distribution[(weather, temperature)] = (
                joint_distribution[temperature, weather]
                / temperature_distribution[temperature]
            )

    return conditional_distribution


if __name__ == "__main__":
    joint_distribution = {
        (Temperature.HOT, Weather.SUN): 0.70,
        (Temperature.HOT, Weather.RAIN): 0.10,
        (Temperature.COLD, Weather.SUN): 0.05,
        (Temperature.COLD, Weather.RAIN): 0.15,
    }

    temperature_distribution = get_temperature_distribution(joint_distribution)
    weather_distribution = get_weather_distribution(joint_distribution)
    conditional_distribution = get_conditional_distribution(joint_distribution)

    def test_for_indepedent_event():
        equation_1 = "p(hot, rain) ?= p(hot) * p(rain)"
        prob_hot_times_rain = (
            temperature_distribution[Temperature.HOT]
            * weather_distribution[Weather.RAIN]
        )
        print(
            f"{equation_1} AS {joint_distribution[Temperature.HOT, Weather.RAIN]} < {prob_hot_times_rain} "
        )

        equation_2 = "p(cold, rain) ?= p(cold) * p(rain)"
        prob_hot_times_rain = (
            temperature_distribution[Temperature.COLD]
            * weather_distribution[Weather.RAIN]
        )
        print(
            f"{equation_2} AS {joint_distribution[Temperature.COLD, Weather.RAIN]} > {prob_hot_times_rain}"
        )

    def test_for_chain_rule():
        # chain rule: P(A, B) = P(A) * P(B | A)
        # decomposed form 1: p(hot, rain) = p(hot) * p(rain | hot)

        chain_rule_1 = "p(hot, rain) = p(hot) * p(rain | hot)"
        prob_hot_rain_chain_rule_1 = (
            temperature_distribution[Temperature.HOT]
            * conditional_distribution[Weather.RAIN, Temperature.HOT]
        )
        print(
            f"{chain_rule_1} AS {joint_distribution[Temperature.HOT, Weather.RAIN]} = {prob_hot_rain_chain_rule_1}"
        )
        # decomposed form 2: p(hot, rain) = p(rain) * p(hot | rain)
        chain_rule_2 = "p(hot, rain) = p(rain) * p(hot | rain)"
        prob_hot_rain_chain_rule_2 = (
            weather_distribution[Weather.RAIN]
            * conditional_distribution[Temperature.HOT, Weather.RAIN]
        )
        print(
            f"{chain_rule_2} AS {joint_distribution[Temperature.HOT, Weather.RAIN]} = {prob_hot_rain_chain_rule_2}"
        )

    # run
    test_for_indepedent_event()
    test_for_chain_rule()
