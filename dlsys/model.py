from typing import Any, List, Tuple, Union

import numpy as np
from nptyping import NDArray

Buffer = NDArray[1, Any]


class FIFOMemory(object):
    def __init__(self, memsize: int):
        self.__memsize = memsize
        self.__usage = 0
        self.__buffer: Buffer = np.array([])

    @property
    def memsize(self) -> int:
        return self.__memsize

    @property
    def contents(self) -> Buffer:
        return self.__buffer

    def add(self, new_val: Any):
        if self.__usage >= self.__memsize:
            self.__buffer = np.delete(self.__buffer, 0)
            self.__usage -= 1
        self.__buffer = np.append(self.__buffer, new_val)
        self.__usage += 1


class DualSysyem(object):
    def __init__(self, theta: float, alpha_p: float, alpha_n: float,
                 tau: float, I: float, C: float, memsize: int,
                 cycle_length: int):
        # model parameters
        self.theta = theta
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.tau = tau
        self.I = I
        self.C = C
        self.__memisize = memsize
        self.__cycle_lenght = cycle_length

        # hidden variables
        self.__rk = 0.
        self.__rk_bar = 0.
        self.__gk = 0.
        self.__hkt = 0.
        self.__hk = 0.

        # memories
        self.__behavior_memory = FIFOMemory(memsize)
        self.__reward_memory = FIFOMemory(memsize)
        # Used in the second term of the right-hand side of Eq. 6
        self.__hkt_memory = FIFOMemory(cycle_length)

        self.__k = -memsize + 1
        self.__t = 0.
        self.__response_count = 0
        self.__reward_count = 0

    @property
    def k(self) -> int:
        if self.__k < 0:
            return 0
        return self.__k

    @property
    def rk(self) -> float:
        return self.__rk

    @property
    def rk_bar(self) -> float:
        return self.__rk_bar

    @property
    def gk(self) -> float:
        return self.__gk

    @property
    def hkt(self) -> float:
        return self.__hkt

    @property
    def hk(self) -> float:
        return self.__hk

    @property
    def t(self) -> float:
        return self.__t

    def update_memory(self, b: int, r: int):
        self.__behavior_memory.add(b)
        self.__reward_memory.add(r)
        self.__k += 1

    def update_rk(self):
        # eq. 1
        b = self.__behavior_memory.contents
        r = self.__reward_memory.contents
        b_bar = np.mean(b)
        r_bar = np.mean(r)
        b_sd = np.std(b)
        r_sd = np.std(r)
        n = self.__memisize
        rk = np.sum((b - b_bar) * (r - r_bar)) / (n * b_sd * r_sd)
        if np.isnan(rk):
            rk = 0.
        self.__rk = rk
        return rk

    def update_rk_bar(self):
        # eq. 2
        if self.__k > 0:
            beta = 1 / self.__k
            self.__rk_bar += beta * (self.__rk - self.__rk_bar)

    def update_gk(self):
        # eq. 3
        rk = self.__rk
        rk_bar = self.__rk_bar
        self.__gk = self.theta * rk + (1 - self.theta) * rk_bar

    def update_hkt(self, PE: float):
        # eq. 4
        alpha = self.alpha_n
        if PE > 0:
            alpha = self.alpha_p
        self.__hkt += alpha * PE
        self.__hkt_memory.add(self.__hkt)

    def compute_prediction_error(self, reward: float) -> float:
        # eq. 5
        if reward > 0:
            self.__reward_count += 1
        return reward - (self.__hkt + self.__gk)

    def update_hk(self):
        # eq. 6
        # the implementation was made to fit the expected model behavior
        # since `h_k` diverges when implemented as described in the paper.
        # following two lines are the implementation as described in the paper.
        # sum_hkt_in_cycle = np.sum(self.__hkt_memory.contents)
        # self.__hk += sum_hkt_in_cycle

        # this implementation shows the expected behavior
        self.__hk = self.__hkt

    def compute_response_probability(self) -> float:
        p = 1 / (1 + np.exp(-self.tau *
                            (self.I * self.__gk + self.__hk - self.C)))
        return p

    def emit_response(self, p: float) -> bool:
        response = np.random.uniform() <= p
        self.__response_count += response
        return response

    def step(self, step_size: float):
        self.__t += step_size
        if self.__t >= self.__cycle_lenght:
            b, r = self.__response_count, self.__reward_count
            self.update_memory(b, r)
            self.update_rk()
            self.update_rk_bar()
            self.update_gk()
            self.update_hk()
            self.__response_count = 0
            self.__reward_count = 0
            self.__t = 0.
