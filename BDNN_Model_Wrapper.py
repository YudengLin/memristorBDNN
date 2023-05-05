from __future__ import division, print_function
import math
import torch
from torch.autograd import Variable
import scipy.io as scio
from scipy import interpolate
import numpy as np
from scipy.ndimage.filters import gaussian_filter
#%%

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out

def LogSumExp(x, dim = None):
    x_max,_ = torch.max(x, dim =dim, keepdim = True)
    return ((x - x_max).exp()).sum(dim=dim,keepdim=True).log()+ x_max


def log_likelihood_values(outputs, target, sample, noise_rhos, location = 0.0, scale = 1.0, reduction=True):
    noise_variance = (torch.exp(noise_rhos)* scale**2)
    targets = target.unsqueeze(1).unsqueeze(1).expand(-1, sample, -1, -1)
    ll =  -0.5 * torch.log(2 * math.pi* noise_variance) - \
           0.5 * (outputs * scale + location - targets) ** 2 / noise_variance
    return torch.sum(ll)

def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)
    sum_tmp = -(log_coeff + exponent)
    if sum_reduce:
        return sum_tmp.sum()
    else:
        return sum_tmp

# %%

class BBP_Model_Wrapper:
    def __init__(self, network, learn_rate, train_set, batch_size, no_batches,risk_path=None):

        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches
        self.X, self.Y = train_set.data, train_set.target
        self.normalize(self.X, self.Y)

        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # self.loss_func = log_gaussian_loss
        self.loss_func =log_likelihood_values

        # load uncert value
        if risk_path is not None:
            self.load_uncert_value(risk_path)

    def normalize(self, X, Y):
        self.std_X = torch.std(X, dim=0)
        self.std_X[self.std_X == 0] = 1.
        self.mean_X = torch.mean(X, dim=0)

        self.std_Y = torch.std(Y, dim=0)
        self.std_Y[self.std_Y == 0] = 1.
        self.mean_Y = torch.mean(Y, dim=0)
        X_norm = (X - self.mean_X) / self.std_X
        Y_norm = (Y - self.mean_Y) / self.std_Y
        return (X_norm, Y_norm)

    def fit(self, x, y, beta,alfa, no_sample,ind=None):
        # reset gradient
        self.optimizer.zero_grad()
        # normalize x and y
        x = (x - self.mean_X) / self.std_X
        y = (y - self.mean_Y) / self.std_Y
        # predict
        outputs, KL_loss_total, _= self.network(x, no_sample, ind)
        # calc loss function
        fit_loss_total = self.loss_func(outputs, y, no_sample, self.network.log_v_noise)
        total_loss = beta*KL_loss_total - fit_loss_total

        # backward and update weight
        total_loss.backward()
        self.optimizer.step()
        # clip window
        self.network.clip_win()

        return total_loss, self.network.z_rhos, self.network.log_v_noise


    def reward(self,state):
        # state: [Batch * N * 1 * state_features]
        x = state[:,:,0]
        y = state[:,:,1]

        # reward, cost: [Batch * N * 1]
        game_over = torch.zeros_like(y)
        game_over[y<0.] = 1.

        distance = torch.sqrt(torch.sum(state * state,dim=2))
        reward = -torch.ones_like(y)*10
        reward[y>=0.] = 20 * (1. - torch.exp(-1. / (distance[y>=0.])))
        reward[reward == float('inf')] = 1000.

        cost = torch.exp(-reward)

        return reward, game_over, cost

    def predict_one_step(self, state, action, no_sample, ext_noises):
        with torch.no_grad():
            output_noise, input_z = ext_noises
            # state, action: [Batch *1 *_]
            x_test = torch.cat((state, action), dim=2)
            x_test = (x_test - self.mean_X) / self.std_X
            mt, _, ave_fly_time = self.network(x_test, no_sample, input_z=input_z)
            vt =  torch.exp(self.network.log_v_noise)

            # mt: mean of output; vt: var of output : [Batch * 1 * output_dim]
            mt = mt * self.std_Y + self.mean_Y
            vt *= self.std_Y ** 2
            # sample from output noise
            # output_noise should be : [Batch * 1 * output_dim]
            output_noise = output_noise

            dstate_add_noise = output_noise * torch.sqrt(vt) + mt
            # obtain state(t+1) by adding dstate_add_noise(t+1): [Batch * N * 1 * output_dim]
            state_t1 = state + dstate_add_noise
            # get cost:  [Batch * N * 1]
            reward_t1, game_over_t1, cost_t1 = self.reward(state_t1)

            return state_t1, reward_t1, game_over_t1, cost_t1, ave_fly_time

    def predict_MC(self,state, action, M ,N ):
        '''
        Monte Carlo approximation:
        For each data sample in a batch of init_state:
        Latent input Z is sampled form P_prior(z) a total of M times,
        and then for each of these samples, N roll-outs are performed
        with Z fixed and sampling only the weight of BNN. And each
        roll-out is as a state_next with one total_step prediction.
        '''

        # state: [Batch *1 *state_features] * weight: [no_sample * state_features *feature_out]
        # note that: Batch = no_sample
        batch = state.size()[0]
        # rewards: [Batch *M *N *1]
        rewards = torch.zeros((batch, M, N, 1)).cuda()
        game_over = torch.zeros((batch, M, N, 1)).cuda()
        state_next = torch.zeros((batch, M, N, 1, state.size()[2])).cuda()
        ave_fly_times = torch.zeros((batch, M, N, 1)).cuda()
        for m in range(M):
            print('m = %d'%(m))
            # input_z_m: [Batch *1 *_]
            input_z_m = (torch.sqrt(self.network.v_prior_z) * torch.randn((batch, 1, 1))).cuda()
            output_noise = torch.randn((batch, 1, self.network.output_dim)).cuda()

            for n in range(N):
                print('\tn = %d' % (n))

                # predict next state state_t1: [Batch *1 *state_features]
                # and cost_t1: [Batch * N * 1]
                ext_noises = [output_noise, input_z_m]
                state_t1, reward_t1, game_over_t1, cost_t1, ave_fly_time  = self.predict_one_step(state, action, batch, ext_noises)
                # save rewards
                rewards[:, m, n] = reward_t1
                game_over[:, m, n] = game_over_t1
                state_next[:, m, n, :, :] = state_t1
                ave_fly_times[:, m, n] = torch.tensor(ave_fly_time)
        return state_next, rewards, game_over, ave_fly_times

    def load_uncert_value(self, path):
        uncert_data = scio.loadmat(path)
        x, y, risk_epistemic, risk_aleatoric = uncert_data['x'], uncert_data['y'], uncert_data['risk_epistemic'], uncert_data['risk_aleatoric']

        risk_epistemic = gaussian_filter(risk_epistemic, sigma=2)
        risk_aleatoric = gaussian_filter(risk_aleatoric, sigma=2)

        self.risk_epistemic_interpfunc = interpolate.interp2d(x, y, risk_epistemic, kind='linear')
        self.risk_aleatoric_interpfunc = interpolate.interp2d(x, y, risk_aleatoric, kind='linear')

    def get_risk(self,state):
        x = state[0]
        y = state[1]
        risk_epistemic = self.risk_epistemic_interpfunc(x, y)
        risk_aleatoric = self.risk_aleatoric_interpfunc(x, y)
        return risk_epistemic, risk_aleatoric

    def decompose_uncert(self, reward, beta, gamma):
        # reward: [Batch * M * N ]

        # average reward: [Batch *1]
        # reward_ave = torch.mean(torch.mean(reward, dim=1), dim=1)

        # risk of aleatoric: [Batch *1]
        risk_aleatoric = torch.var(torch.mean(reward, dim=2), dim=1)  
        # risk of epistemic: [Batch *1]
        risk_epistemic = torch.mean(torch.var(reward, dim=2), dim=1)

        return risk_aleatoric, risk_epistemic
