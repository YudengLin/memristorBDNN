from __future__ import division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from Dataset.mydataset import Dataset
import argparse
import numpy as np
from BDNN_Model import BBP_Model
from BDNN_Model_Wrapper import BBP_Model_Wrapper
import time
import torch.utils.data

#%%
def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# %% args
parser = argparse.ArgumentParser(description='Train Bayesian Neural Net on Wet-chicken with Variational Inference')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Local_Reparam',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=3000,
                    help='How many epochs to train. Default: 3000.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-4.')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=5,
                    help='How many MC samples to take when approximating the ELBO. Default: 3.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BDNN_models',
                    help='Where to save learnt weights and train vectors. Default: \'BDNN_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='BDNN_results',
                    help='Where to save learnt training plots. Default: \'BDNN_results\'.')
args = parser.parse_args()

# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir
mkdir(models_dir)
mkdir(results_dir)
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
# warnings.filterwarnings('ignore')
# %% dataset
dirs = './Dataset/out/2D/'
print('\nData: '+dirs)
# load data
mytrainset = Dataset(root=dirs, train=True, to_tensor=True, preprocess_data = False)
trainset = torch.utils.data.TensorDataset(mytrainset.data, mytrainset.target)
mytestset = Dataset(root=dirs, train=False, to_tensor=True, preprocess_data = False)
testset = torch.utils.data.TensorDataset(mytestset.data, mytestset.target)

batch_size = 200
NTrainPoints = mytrainset.len()
nb_train =int(NTrainPoints / batch_size)

#%% define BNN+LV
no_sample = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration

output_dim = 2
input_dim = 4
layer_size = [input_dim, 100, 100, output_dim]
learn_rate = args.lr

# SimulationMode:
#   True    - use memristor simulation model
#   False   - use ESCIM computing platform
# Here, we train world model to get memristor weight targets, so
# 'SimulationMode' has to be set to 'True'.
Exp = {'SimulationMode':True}
print('\nNetwork:')
net = BBP_Model(layer_size, NTrainPoints, no_sample, Exp).cuda()
net.train(True)
world_model = BBP_Model_Wrapper(net, learn_rate, mytrainset,batch_size, nb_train)

#%% main train
num_epochs=int(args.epochs)
log_every=10

total_loss = np.zeros(num_epochs)
best_loss = float('inf')
best_net = type(net)(layer_size, NTrainPoints, no_sample, Exp)  # get a new instance

print('Start epochs!')
for i in range(num_epochs):
    tic = time.time()
    inds = np.random.permutation(NTrainPoints)
    for  k in range(nb_train):
        ind = inds[k*batch_size : (k+1)*batch_size]
        x, y = trainset[ind]
        beta = 2 ** (nb_train - (k + 1)) / (2 ** nb_train - 1)
        alfa = 10
        # nn fit
        loss, z_rhos, log_v_noise = world_model.fit(x, y, beta,alfa, no_sample, ind)

        total_loss[i] = loss.cpu().data.numpy()
        if total_loss[i] < best_loss:
            best_loss = total_loss[i]
            best_net.load_state_dict(world_model.network.state_dict()) # copy weights and stuff

    toc = time.time()
    if i%1 == 0:
        print('Epoch: %s/%d, Train loss = %.3f z_rhos = %.3f noise_rhos = %.3f' % \
              (str(i + 1).zfill(3), num_epochs, total_loss[i], z_rhos.mean(), log_v_noise.mean()))

filename = 'theta_best.dat'
save_path = models_dir +'/'+ filename
torch.save(best_net.state_dict(), save_path)
print('save net in ' + save_path)

