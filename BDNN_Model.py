from __future__ import division, print_function
from torch.autograd import Variable
from memristor_model import *
from gaussian import gaussian
# %%

class BBP_Model(nn.Module):
    def __init__(self, layer_size, N_train, no_sample=None, Exp=None):
        super(BBP_Model, self).__init__()
        input_dim = layer_size[0] + 1 #add 1 input dim for hidden input Z
        self.input_dim = input_dim
        self.output_dim = layer_size[-1]
        self.layer_size = [input_dim] + layer_size[1:]
        self.N_train = N_train
        print(self.layer_size)

        self.init_states = [[1.4,2.4,3.2],[1.4,2.4,3.2],[1.4,2.4,3.2]]
        # N: number of device in each weight
        self.N = [3, 3, 3]
        # offset, scale can be tuned manually according to specified task
        self.offset = [7.2, 7.2, 7.2]
        self.scale = [10.5, 10.5, 10.5]
        self.gaussian_var = [0.01,0.01,0.01]
        self.no_sample = no_sample
        self.update_fail_sum = 0

        # hidden input Z
        self.z_mus = nn.Parameter(torch.Tensor(self.N_train,1).uniform_(0.9, 1.1))
        self.z_rhos = nn.Parameter(torch.Tensor(self.N_train,1).uniform_(5.0, 5.5))
        self.v_prior_z = torch.tensor(input_dim-1,dtype=torch.float)
        self.z_prior = gaussian(0, self.v_prior_z)

        # network with two hidden and one output layer
        # self.layer_size = [input_dim, num_units, num_units, output_dim]
        self.num_weight_layer = len(self.layer_size) - 2
        self.layers = nn.ModuleList()
        for layer_id, (in_d, ou_d) in enumerate(zip(self.layer_size[:-1], self.layer_size[1:])):
            layer = BayesLinear_memristorLayer(layer_id, in_d, ou_d, gaussian(0, self.gaussian_var[layer_id]), self.N[layer_id], self.offset[layer_id],
                                          self.scale[layer_id], self.init_states[layer_id], no_sample, Exp)
            self.layers.append(layer)

        # activation to be used between hidden XBArray
        self.activation = nn.ReLU(inplace=True)
        # output additive noise
        self.log_v_noise = nn.Parameter(torch.Tensor(1, self.output_dim).uniform_(-3, -3))


    def clip_win(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.clip_win()

    def logistic(self, x):
        logi = 1.0 / (1.0 + torch.exp(-x))
        return logi

    def forward(self, x, no_sample, ind=None, input_z=False):
        KL_loss_total = 0.
        if isinstance(input_z, bool):
            # Train BNN+LV mode,input x: [Batch * features]
            x = x.view(-1, x.size()[1])
            # x: [Batch * no_sample * 1 * features]
            x = x.unsqueeze(1).unsqueeze(1).expand(-1, no_sample, -1, -1)
            # calculate the z sample for each x
            # latent input z  : [Batch * no_sample * 1 * 1]
            size = torch.Size([x.size()[0], no_sample, 1, 1])
            z_epsilons = Variable(self.z_mus.data.new(size).normal_())
            z_var = 1e-6 + self.logistic(self.z_rhos[ind]) * (self.v_prior_z - 2e-6)
            z_stds = torch.sqrt(z_var).unsqueeze(1).unsqueeze(1).expand(-1, no_sample, -1, -1)
            z_mus = self.z_mus[ind].unsqueeze(1).unsqueeze(1).expand(-1, no_sample, -1, -1)
            z_sample = z_mus + z_epsilons * z_stds

            # computing the KL loss term of z
            prior_cov, varpost_cov = self.z_prior.sigma ** 2, z_stds ** 2
            KL_loss = 0.5 * (torch.log(prior_cov / varpost_cov)).sum() - 0.5 * z_stds.numel()
            KL_loss = KL_loss + 0.5 * (varpost_cov / prior_cov).sum()
            KL_loss = KL_loss + 0.5 * ((self.z_mus - self.z_prior.mu) ** 2 / prior_cov).sum()
            KL_loss_total += KL_loss

            x = torch.cat((x, z_sample), dim=3)

        elif isinstance(input_z, torch.Tensor):
            # Prediction mode, input x: [Batch * 1 * features]
            # use the input noise externally, ext_noise: [Batch * 1 * 1]
            # z_sample:[Batch * 1 * 1]
            # input_z should be : [Batch * 1 * 1]
            z_sample = input_z
            x = torch.cat((x, z_sample), dim=2)

        pre_scale_factor = 1.
        ave_fly_time = 0.
        for id, layer in enumerate(self.layers):
            x, KL_loss, pre_scale_factor, ave_fly_time_tmp = layer(x, no_sample, pre_scale_factor)
            if ave_fly_time_tmp is not None:
                ave_fly_time = ave_fly_time_tmp

            KL_loss_total += KL_loss
            if id != self.num_weight_layer:
                x = self.activation(x)
            else:
                x = layer.dequant(x)

        return x, KL_loss_total, ave_fly_time



