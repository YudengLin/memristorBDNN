from __future__ import division, print_function
import torch
import torch.nn as nn
import numpy as np

# Simulated memristor device paras
class memristor_paras():
    def __init__(self, layer_id, input_dim, output_dim,N,window_max=4,window_min=0.4,window_flu=False,win_std_flu=0.1,Exp=None):
        # layer paras
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.layer_id = layer_id
        self.Exp = Exp
        # window paras
        # unit: μA
        self.win_std_flu = win_std_flu
        if window_flu:
            self.win_max = (window_max + win_std_flu*torch.randn((input_dim, output_dim,N))).cuda()
            self.win_min = (abs(window_min + 0.1*win_std_flu*torch.randn((input_dim, output_dim,N)))).cuda()
            self.win_ratio = torch.tensor(window_max/window_min).cuda()
            self.win_width = torch.tensor(window_max - window_min).cuda()
        else:
            self.win_max = (window_max + torch.zeros((input_dim, output_dim,N))).cuda()
            self.win_min = (window_min + torch.zeros((input_dim, output_dim,N))).cuda()
            self.win_ratio = torch.tensor(window_max/window_min).cuda()
            self.win_width = torch.tensor(window_max - window_min).cuda()

        # read model paras are fitted under unit μA
        self.a = -0.028
        self.b = 0.14
        self.c = 0.038

    def read_paras(self,state):
        # state, unit: μA
        # Laplace distribution fit paras, modeling in unit μA
        scale = -0.0058*state.pow(2) + 0.0324*state + 0.0141
        mus = torch.zeros_like(scale)

        return mus, scale

    def read_noise(self, state):
        # unit: μA
        # get scale factor by Laplace fit paras
        read_mus, read_scale = self.read_paras(state)
        Laplace = np.random.laplace(loc=read_mus.detach().cpu().numpy(), scale=read_scale.detach().cpu().numpy(), size=None)
        read_noise = torch.tensor(Laplace.astype(np.float32)).cuda()

        return read_noise

# Simulated memristor device
class memristor_device(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim,N=3,init_states=[1.4,2.4,3.2],Exp=None):
        super(memristor_device, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.N = N
        self.layer_id = layer_id
        self.paras = memristor_paras(layer_id, input_dim, output_dim,N,window_max=4,window_min=0.4,window_flu=True,win_std_flu=0.2,Exp=Exp)
        # device state now, unit μA
        init_current = torch.empty(input_dim, output_dim, N)
        for idx,init_state in enumerate(init_states):
            init_current[:,:,idx].uniform_(init_state - 0.7, init_state + 0.7)
            # print('Init uniform current state...')

        self.current_state = nn.Parameter(init_current)

    def sample(self,no_sample):
        # each weight = sum read current of N memristor devices
        # here we sample each weight for no_sample times
        read_sum_N = torch.zeros((no_sample,self.input_dim,self.output_dim)).cuda()
        for s in range(no_sample):
            read_Ndevice = self.read(self.current_state)
            read_sum_N[s,:,:] = torch.sum(read_Ndevice,dim=2).cuda()
        read_sum_N_mus = torch.mean(read_sum_N,dim=0).cuda()
        read_sum_N_std = torch.std(read_sum_N,dim=0).cuda()

        # return samples, mus and std
        # v=std^2, v(x1+x2+...)=v(x1)+v(x2)+...
        return read_sum_N,read_sum_N_mus,read_sum_N_std # [no_sample * input_dim * output_dim]

    def read(self,state):
        # state: current state of each memristor device, [input_dim, output_dim, N]
        # get read noise based on memristor model paras
        read_noise = self.paras.read_noise(state)
        # add to current state
        read_out = state + read_noise
        return read_out # [input_dim, output_dim, N]

    def clip_win(self):
        state = self.current_state.data
        # window clip
        win_min = self.paras.win_min
        win_max = self.paras.win_max
        min_overflow = state < win_min
        max_overflow = state > win_max
        state[min_overflow] = win_min[min_overflow]
        state[max_overflow] = win_max[max_overflow]
        self.current_state.set_(state)

# Hardware platform device
class Platform_device():
    def __init__(self, layer_id, input_dim, output_dim, N=3, Exp=None):
        print('Please complete platform code.')
        exit()        

    def sample(self,no_sample):
        print('Please complete platform code.')
        exit()        
        # This is return samples, mus and std
        # return read_sum_N,read_sum_N_mus,read_sum_N_std, ave_fly_time

class XBArray(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, N=3, init_states=[1.4,2.4,3.2],offset=7.2, scale=1.,no_sample=5, Exp=None):
        super().__init__()
        # input_dim += 1  # add 1 for bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_id = layer_id
        self.devices = []
        self.N, self.offset, self.scale, self.no_sample = N, offset, scale, no_sample
        # SimulationMode: True-use memristor paras model ; False- use platform
        self.Simulation_Mode = Exp['SimulationMode']
        if Exp['SimulationMode']:
            self.devices = memristor_device(layer_id, input_dim, output_dim, N, init_states, Exp)
        elif not Exp['SimulationMode']:
            self.XBArray_plat = Platform_device(layer_id, input_dim, output_dim, N, Exp)
            self.weight = nn.Parameter(torch.ones(no_sample, input_dim, output_dim))

    def sample(self, no_sample=None):
        if no_sample == None:
            no_sample = self.no_sample
        # sample weight
        ave_fly_time = 0.
        if self.Simulation_Mode:
            raw_weight_sample, raw_weight_mus, raw_weight_stds = self.devices.sample(no_sample)

        elif not self.Simulation_Mode:
            raw_weight_sample, raw_weight_mus, raw_weight_stds, ave_fly_time = self.XBArray_plat.sample(no_sample)

        weight_sample = ((raw_weight_sample - self.offset)/self.scale)
        weight_mus = ((raw_weight_mus - self.offset)/self.scale)
        weight_stds = (raw_weight_stds / self.scale)

        # This function returns weight samples, mus and std, and (ave_fly_time of platform)
        return weight_sample, weight_mus, weight_stds, ave_fly_time

    def clip_win(self):
        if self.Simulation_Mode:
            self.devices.clip_win()
        elif not self.Simulation_Mode:
            return

    def forward(self, x, no_sample=None):
        # sample gaussian noise for each weight and each bias
        weight_sample, self.weight_mus, self.weight_stds, ave_fly_time = self.sample(no_sample)
        if self.Simulation_Mode:
            output = torch.matmul(x, weight_sample)
        elif not self.Simulation_Mode:
            with torch.no_grad():
                self.weight.set_(weight_sample)
            output = torch.matmul(x, self.weight)

        return output, self.weight_mus, self.weight_stds, ave_fly_time

class BayesLinear_memristorLayer(nn.Module):
    def __init__(self, layer_id, input_dim, output_dim, prior, N, offset, scale,init_states, no_sample, Exp):
        super(BayesLinear_memristorLayer, self).__init__()
        input_dim = input_dim + 1 # for bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior = prior
        self.layer_id = layer_id
        self.N, self.offset, self.scale, self.no_sample = N, offset, scale, no_sample
        self.init_states = init_states
        self.XBAccelerator = XBArray(layer_id, input_dim, output_dim, N, init_states, offset, scale, no_sample, Exp)

        # calc_quant_scale
        self.calc_quant_scale = False
        # history of max x
        self.x_max_pre = torch.zeros(1).cuda()
        self.quant_scale = nn.Parameter(torch.ones([]), requires_grad=False)
        self.acc_scale = 1.
        if layer_id ==0 or layer_id==2:
            self.quant_min, self.quant_max = -127, 127
        else:
            self.quant_min, self.quant_max = 0, 255
        self.cut_factor = 1

    def clip_win(self):
        self.XBAccelerator.clip_win()

    def quant(self, x, pre_scale_factor=1.):
        self.acc_scale = pre_scale_factor * self.quant_scale

        bias = 1.
        if self.XBAccelerator.training:
            # training mode
            return x, bias
        else:
        # when not training, we get the scale param of quantization or x is quantized
            if self.calc_quant_scale :
                # calc scale mode
                x_max = torch.max(abs(x.flatten()))
                if self.x_max_pre<x_max:
                    self.x_max_pre = x_max
                    # x*s = xq  s=xq/x
                    self.quant_scale.set_(((self.cut_factor * self.quant_max) / self.x_max_pre).cuda())
                    print('Layer ID %d quant_scale= %f' % (self.layer_id, self.quant_scale.cpu().numpy()))
                return x, bias
            else:
                if self.quant_scale==1. and pre_scale_factor==1.:
                    return x, bias
                else:
                    # inference mode with quantized x
                    # we convert input to quantized x: xq=x*s
                    x = (x * self.quant_scale).round().clamp_(self.quant_min,self.quant_max)
                    bias = (self.acc_scale).round().clamp_(self.quant_min,self.quant_max)
                    return x, bias

    def dequant(self,x):
        if self.XBAccelerator.training:
            # training mode
            return x
        else:
            # inference mode
            if self.calc_quant_scale:
                # calc scale mode
                return x
            else:
                # inference mode with quantized x
                return  (x/(self.acc_scale))

    def forward(self, x, no_sample=None, pre_scale_factor=1.):

        x, bias = self.quant(x, pre_scale_factor)
        # input x now concentrates 1. as bias input
        dims = [dim for dim in x.size()[0:-1]]
        dims.append(1)
        x = torch.cat((x, (torch.ones(dims)*bias).cuda()), len(dims)-1)

        output, weight_mus, weight_stds, ave_fly_time = self.XBAccelerator(x, no_sample)

        # computing the KL loss term
        prior_cov, varpost_cov = self.prior.sigma ** 2, weight_stds ** 2
        KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * weight_stds.numel()
        KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
        KL_loss = KL_loss + 0.5 * ((weight_mus - self.prior.mu) ** 2 / prior_cov).sum()

        return output, KL_loss, self.acc_scale, ave_fly_time
