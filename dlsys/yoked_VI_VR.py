from typing import List, Tuple

import numpy as np
from nptyping import NDArray
from pandas import DataFrame
from scipy.stats import expon

from dlsys.model import DualSysyem


def expon_equally_spaced(mean_interval: float, _min: float,
                         n: int) -> NDArray[1, float]:
    intervals = expon.ppf(
        np.linspace(0.01, 0.99, n), scale=mean_interval, loc=_min) - _min
    return np.random.choice(intervals, size=len(intervals), replace=False)


def update_agent_state(agent: DualSysyem, b: int, r: int):
    agent.update_memory(b, r)
    agent.update_rk()
    agent.update_rk_bar()
    agent.update_gk()
    agent.update_hk()


RowOfResult = Tuple[float, float]  # gk, hk


def variable_interval_schedule(
        agent: DualSysyem,
        intervals: NDArray[1, float]) -> Tuple[List[int], DataFrame]:
    # for yoked control
    response_since_reward = 0
    required_responses: List[int] = []

    row_of_result: List[RowOfResult] = []

    for interval in intervals:
        response = False
        while interval > 0 or not response:
            p = agent.compute_response_probability() * STEP_SIZE
            response = agent.emit_response(p)
            interval -= STEP_SIZE

            response_since_reward += response

            if interval <= 0. and response:
                rpe = agent.compute_prediction_error(1.)

                required_responses.append(response_since_reward)
                response_since_reward = 0
            elif response:
                rpe = agent.compute_prediction_error(0.)
            else:
                rpe = 0.
            agent.update_hkt(rpe)
            gk_hk = agent.step(STEP_SIZE)
            if gk_hk is not None:
                row_of_result.append(gk_hk)

    result = DataFrame(row_of_result, columns=["gk", "hk"])
    return required_responses, result


def variable_ratio_schedule(agent: DualSysyem,
                            required_responses: List[int]) -> DataFrame:
    row_of_result: List[RowOfResult] = []

    for required_response in required_responses:
        while required_response > 0:
            p = agent.compute_response_probability() * STEP_SIZE
            response = agent.emit_response(p)

            required_response -= response

            if required_response <= 0 and response:
                rpe = agent.compute_prediction_error(1.)
            elif response:
                rpe = agent.compute_prediction_error(0.)
            else:
                rpe = 0.
            agent.update_hkt(rpe)
            gk_hk = agent.step(STEP_SIZE)
            if gk_hk is not None:
                row_of_result.append(gk_hk)

    result = DataFrame(row_of_result, columns=["gk", "hk"])
    return result


if __name__ == '__main__':
    from pathlib import Path

    from dlsys.model import DualSysyem

    # parameters for simulation
    NUM_SUBJECTS = 5
    STEP_SIZE = 1 / 3
    NUMBER_OF_TRIAL = 30
    SESSION = 10
    MEAN_INTERVAL = 15

    # model parameters
    THETA = .1
    ALPHAP = 2 * 10**-2
    ALPHAN = 1 * 10**-5
    TAU = 10.
    I = 1.
    C = .6
    MEMSIZE = 5
    STEP_IN_CYCLE = 20

    merged_data = DataFrame()

    for agt_id in range(NUM_SUBJECTS):
        agent_vi = DualSysyem(THETA, ALPHAP, ALPHAN, TAU, I, C, MEMSIZE,
                              STEP_IN_CYCLE)
        agent_vr = DualSysyem(THETA, ALPHAP, ALPHAN, TAU, I, C, MEMSIZE,
                              STEP_IN_CYCLE)

        intervals = expon_equally_spaced(MEAN_INTERVAL, 0.1, NUMBER_OF_TRIAL)
        for session in range(SESSION):
            required_responses, result_vi = variable_interval_schedule(
                agent_vi, intervals)
            result_vr = variable_ratio_schedule(agent_vr, required_responses)
            result_vi["subject"] = agt_id
            result_vr["subject"] = agt_id
            result_vi["condition"] = "VI"
            result_vr["condition"] = "VR"
            result_vi["session"] = session
            result_vr["session"] = session

            merged_data = merged_data.append([result_vi, result_vr])

    project_root: Path = Path(__file__).absolute().parent.parent
    data_dir = project_root.joinpath("data")
    if not data_dir.exists():
        data_dir.mkdir()
    filename = data_dir.joinpath("yoked_VI_VR.csv")
    merged_data.to_csv(filename, index=False)
