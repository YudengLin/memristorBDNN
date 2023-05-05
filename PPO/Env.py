import os
from os import path
import sys

d = path.dirname(__file__) 
parent_path = os.path.dirname(d)  
sys.path.append(parent_path)
from BDNN_Model import BBP_Model
from BDNN_Model_Wrapper import BBP_Model_Wrapper
from Dataset.mydataset import Dataset
import torch
from Dataset.Environment import Environment
import numpy as np
import gym
from gym import spaces


def make_env(Exp, file_name, risk_beta=0.,risk_gamma=0.,init_state=None):
    if  Exp['Env'] == 0:
        env = Environment()
    else:
        env = Virtual_Environment(file_name, Exp, risk_beta, risk_gamma)

    return env


class Virtual_Environment(gym.Env):
    def __init__(self, file_name, Exp, risk_beta=0.,risk_gamma=0.,init_state=None):
        super(Virtual_Environment, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]),
                                            dtype=np.float64)
        if init_state is None:
            self.init_state = np.array([20., 10.])
        else:
            self.init_state = init_state
        self.Exp = Exp
        self.risk_beta, self.risk_gamma = risk_beta, risk_gamma
        # load data
        dirs = '../Dataset/out/2D/'
        mytrainset = Dataset(root=dirs)
        state_dim = 2
        action_dim = 2
        state_start = mytrainset.data[:, :state_dim].cpu().numpy()
        batch_size = 200
        NTrainPoints = mytrainset.len()
        nb_train = int(NTrainPoints / batch_size)

        # World model
        out_dim = state_dim
        in_dim = state_dim + action_dim
        num_units = 100
        learn_rate = 0.01
        batch_size = 100
        self.no_sample = 1
        layer_size = [in_dim, num_units, num_units, out_dim]
        self.model = BBP_Model(layer_size, NTrainPoints, self.no_sample, Exp).cuda()

        PATH = '../BDNN_models/'
        print('\nWorld model:' + file_name)
        if Exp['SimulationMode']:
            self.model.load_state_dict(torch.load(PATH + file_name))
        else:
            # use ESCIM platform, and  will load quant scale paras
            # please set 'quant_model_file_name' as 'file_name'
            self.model.load_state_dict(torch.load(PATH + file_name), strict=False)
        self.model.eval()
        risk_path = '../Uncert_decom_data/uncert_decom_result.mat'
        self.world_model = BBP_Model_Wrapper(self.model, learn_rate, mytrainset, batch_size, nb_train, risk_path)

    def reset(self, clear_state_trans=False):
        self.state = self.init_state
        self.t, self.T = 0, 100
        if clear_state_trans:
            self.clear_state_trans()
        return self.get_state()

    def clear_state_trans(self):
        self.state_trans = []

    def get_state(self):
        return self.state

    def render(self, mode='human_virtual'):
        pass

    def step(self, action):
        # predict next state by world model without uncertainty decompose
        input_z = (torch.randn((self.no_sample, 1, 1)) * torch.sqrt(self.world_model.network.v_prior_z)).cuda()
        output_noise = torch.randn((self.no_sample, 1, self.world_model.network.output_dim)).cuda()
        ext_noises = [output_noise, input_z]
        # state, not included Action: [Batch *state_features]
        state = torch.tensor(self.state.astype(np.float32))
        action = torch.tensor(action.astype(np.float32))
        # state: [Batch *1 *state_features]
        state = state.unsqueeze(0).expand(self.no_sample, -1, -1).cuda()
        action = action.unsqueeze(0).expand(self.no_sample, -1, -1).cuda()
        next_state, reward_t1, game_over, _, fly_time = self.world_model.predict(state, action, self.no_sample,
                                                                                 ext_noises)
        reward_t1 = reward_t1.cpu().numpy()
        if self.Exp['Env'] == 1:
            risk_epistemic, risk_aleatoric = self.world_model.get_risk(self.state)
            reward_t1 = reward_t1 - self.risk_beta * risk_epistemic - self.risk_gamma * risk_aleatoric

        self.state = next_state[0,0].cpu().numpy()
        reward = reward_t1[0, 0]

        if self.t >= self.T:
            done = True
        else:
            if game_over[0, 0].cpu().numpy()==1.:
                done = True
            else:
                done = False
        info = {}

        self.t += 1
        return self.state, reward, done, info

