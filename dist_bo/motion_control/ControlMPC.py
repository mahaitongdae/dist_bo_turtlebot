"""
    <Reinforcement Learning and Control>(Year 2020)
    by Shengbo Eben Li
        @ Intelligent Driving Lab, Tsinghua University

    OCP example for lane keeping problem in a circle road

    [Method]
    Model predictive control

"""
from casadi import *
# from config import DynamicsConfig
import math
# from dynamics import VehicleDynamics
import matplotlib.pyplot as plt
import numpy as np

LBX = -3.
UBX = 0.
LBY = 0.
UBY = 4.
ROBOT_RADIUS = 0.11
ROBOT_SAFE_RADIUS = 1.2 * ROBOT_RADIUS
DYNAMICS_DIM = 9
ACTION_DIM = 6

class MPCSolver(object):
    def __init__(self):
        self.agent_num = 3

    def solve(self, init_loc, target_loc, predict_steps = 10):
        x = SX.sym('x', DYNAMICS_DIM)
        u = SX.sym('u', ACTION_DIM)

        # Create solver instance
        ref_a_b = []
        for i in range(self.agent_num):
            ref_a_b += [(x[3*i+2] - x[3*i])]

        # x : [x1, x2, v, theta]
        f = vertcat(
            x[0] + self.Ts * u[0] * cos(x[2]),
            x[1] + self.Ts * u[0] * sin(x[2]),
            x[2] + self.Ts * u[1],
            x[3] + self.Ts * u[2] * cos(x[5]),
            x[4] + self.Ts * u[2] * sin(x[5]),
            x[5] + self.Ts * u[3],
            x[6] + self.Ts * u[4] * cos(x[8]),
            x[7] + self.Ts * u[4] * sin(x[8]),
            x[8] + self.Ts * u[5],

        )
        F = Function("F", [x, u], [f])

        h = vertcat(
            (x[0] - x[3]) ** 2 + (x[1] - x[4]) ** 2 - ROBOT_SAFE_RADIUS ** 2,
            (x[0] - x[6]) ** 2 + (x[1] - x[7]) ** 2 - ROBOT_SAFE_RADIUS ** 2,
            (x[3] - x[6]) ** 2 + (x[4] - x[7]) ** 2 - ROBOT_SAFE_RADIUS ** 2,
        )
        H = Function("H", [x, u], [h])

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', DYNAMICS_DIM)
        w += [Xk]
        lbw += init_loc
        ubw += init_loc

        for k in range(1, predict_steps + 1):
            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, ACTION_DIM)
            w += [Uk]
            lbw += self.U_LOWER
            ubw += self.U_UPPER
            # Gk = self.G_f(Xk,Uk)

            Fk = F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, DYNAMICS_DIM)

            Hk = H(Xk, Uk)

            # Dynamic Constriants
            G += [Fk - Xk]
            lbg += [[0.] * 9]
            ubg += [[0.] * 9]
            G += [Hk]
            lbg += [[0.] * 3]
            ubg += [[inf] * 3]
            w += [Xk]
            for i in range(self.agent_num):
                lbw += [LBX, LBY, -inf]
                ubw += [UBX, UBY, inf]

            # Cost function
            # if tire_model == 'Fiala':
            #
            # else:
            #     F_cost = Function('F_cost', [x, u], [1 * (x[0]) ** 2
            #                                          + 1 * (x[2]) ** 2
            #                                          + 1 * u[0] ** 2])
            F_cost = Function('F_cost', [x, u], [(ref_a_b[0] * x[0] + ref_a_b[1] - x[1]) ** 2
                                                 + (ref_a_b[2] * x[3] + ref_a_b[3] - x[4]) ** 2
                                                 + (ref_a_b[4] * x[6] + ref_a_b[5] - x[7]) ** 2
                                                 + 10 * u[0] ** 2 + 10 * u[1] ** 2])
            J += F_cost(w[k * 2], w[k * 2 - 1])
            # J += F_cost(w[k * 3 - 1], w[k * 3 - 2])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=lbw, ubx=ubw, x0=0, lbg=lbg, ubg=ubg)
        # print(r['x'])
        state_all = np.array(r['x'])
        state = np.zeros([predict_steps, DYNAMICS_DIM])
        control = np.zeros([predict_steps, ACTION_DIM])
        nt = DYNAMICS_DIM + ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(predict_steps):
            state[i] = state_all[nt * i: nt * i + DYNAMICS_DIM].reshape(-1)
            control[i] = state_all[nt * i + DYNAMICS_DIM: nt * (i + 1)].reshape(-1)
        return state, control

def main():
    solver = MPCSolver()
    state, control = solver.solve([0.0,0.0,0.0, 1.0,1.0, 1.0, -1.0,-1.0,-1.0])
