#coding: utf-8
"""
Wet-Chicken benchmark

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import gym
from gym import spaces, logger
from gym.utils import seeding


class Environment(gym.Env):
    ''' Benchmark '''

    def __init__(self, init_state=None,seed=3):
        super(Environment, self).__init__()
        if init_state is None:
            init_state = [20., 10.]
        self.state_dim = 2
        self.action_dim = 2
        self.target_point = np.array([0., 0.])
        self.coastline = 0.
        self.monsoon = np.array([-0.3, -0.15])

        self.init_state = np.array(init_state)
        self.state = self.init_state
        self.game_over = False
        self.state_trans = []
        self.observation_shape = np.shape(self.get_state())[0]
        self.reset_cnt = 0
        self.t = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]), dtype=np.float64)

    def seed(self,seed):
        np.random.seed(seed)

    def reset(self, clear_state_trans=False):
        self.state = self.init_state
        self.t = 0
        self.reset_cnt +=1
        self.clear_state_trans()
        return self.get_state()

    def set(self, state, clear_state_trans=False):
        self.state = state
        if clear_state_trans:
            self.clear_state_trans()
        return self.get_state()

    def clear_state_trans(self):
        self.state_trans = []

    def get_state(self):
        return self.state

    def step(self, action):
        pre_state = self.state
        # get stochastic wave
        wave = self.wave(self.state)
        # calc next state
        self.state = self.state + self.monsoon + action + wave
        # record action and state transition
        self.state_trans.append(np.concatenate((pre_state,action,self.state ),axis=0))
        # get current reward and check if game over
        reward, done = self.reward(self.state, self.t)

        info = {}
        self.t += 1
        return self.state, reward, done, info

    def reward(self,state, t):
        y = state[1]
        x = state[0]
        T = 100
        if y < 0.:
            # game over
            done = True
            reward = -10.
        else:
            if t >= T:
                # end
                done = True
            else:
                done = False
            dxy = state - self.target_point
            dist = np.sqrt(np.sum(dxy*dxy))
            reward = 20*(1.- np.exp(-1./(dist)))

        return reward, done

    def wave(self,state):
        k = -0.18
        b = 0.9
        y = state[1]
        sigma = np.exp(k*y+b)
        wx = 0.
        wy = sigma*np.random.randn(1).squeeze()
        w = np.array([wx, wy])
        return w

    def render(self, mode='human'):
        pass

    def plot(self):
        pass

    def traj_plot(self):
        pass
    # Test
if __name__ == '__main__':
    n_runs_train = 4000
    n_runs_test = 3000
    state_dim = 2
    action_dim = state_dim
    dim = state_dim +action_dim

    X_train = np.zeros((n_runs_train,dim))  # type: ndarray
    Y_train = np.zeros((n_runs_train,state_dim))
    X_test = np.zeros((n_runs_test,dim))
    Y_test = np.zeros((n_runs_test,state_dim))

    # init env
    Env = Environment()
    step_train = 0
    step_test = 0
    warm_up_step = 2
    # get train dataset
    while(step_train < n_runs_train):
        # pick random action
        action = np.random.uniform(-1, 1, 2)
        # one total_step
        s_next, r, dead, s_trans = Env.step(action)
        step_train += 1

    s_trans = np.array(s_trans)
    X_train = s_trans[:,:dim]
    Y_train = s_trans[:,dim:]

    # get test dataset
    Env.reset(clear_state_trans=True)
    while (step_test < n_runs_test):
        # pick random action
        action = np.random.uniform(-1, 1, 2)
        # one total_step
        s_next, r, dead, s_trans = Env.step(action)
        step_test += 1

    s_trans = np.array(s_trans)
    X_test = s_trans[:, :dim]
    Y_test = s_trans[:, dim:]

    plt.scatter(X_test[:, 0], X_test[:, 1],s=2)
    plt.plot(Env.init_state[0], Env.init_state[1], 'ks')
    plt.annotate('  S0', xytext=(Env.init_state[0], Env.init_state[1]), xy=(Env.init_state[0], Env.init_state[1]))
    plt.show()

    dire = './out/2D/'
    isExists = os.path.exists(dire)
    if not isExists:
        os.makedirs(dire)
        print( dire + ' dir created!')
    else:
        print( dire + ' dir existed!')
    np.savetxt(dire + 'X_train.txt', X_train, fmt='%5.4f')
    np.savetxt(dire + 'Y_train.txt', Y_train, fmt='%5.4f')
    np.savetxt(dire + 'X_test.txt', X_test, fmt='%5.4f')
    np.savetxt(dire + 'Y_test.txt', Y_test, fmt='%5.4f')
    print ('Made data over!')
