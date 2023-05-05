import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import scipy.io as scio
from Dataset.mydataset import Dataset
from BDNN_Model import BBP_Model
from BDNN_Model_Wrapper import BBP_Model_Wrapper
import numpy as np

def init_world_model(Exp, file_name):
    #%% dataset
    dirs = './Dataset/out/2D/'
    print('\nData: '+dirs)
    preprocess_data = False

    trainset = Dataset(root=dirs, preprocess_data = preprocess_data)
    batch_size = 200
    NTrainPoints = trainset.len()
    nb_train =int(NTrainPoints / batch_size)
    output_dim = 2
    input_dim = 4
    layer_size = [input_dim, 100, 100, output_dim]
    learn_rate = 0.01
    batch_size = 100
    sample = 1
    model = BBP_Model(layer_size, NTrainPoints,sample, Exp=Exp).cuda()
    world_model = BBP_Model_Wrapper(model, learn_rate, trainset, batch_size, nb_train)

    PATH = './BDNN_models/'
    print('\nModel: ' + file_name)
    if Exp['SimulationMode']:
        model.load_state_dict(torch.load(PATH + file_name))
    else:
        # using plat, and  will load quant scale paras
        model.load_state_dict(torch.load(PATH + file_name), strict=False)
    model.eval()
    return world_model, sample
#%%
def policy(state):
    action = torch.zeros_like(state)
    return action

#
if __name__ == '__main__':
    # SimulationMode:
    #   True    - use memristor simulation model
    #   False   - use ESCIM computing platform
    Exp = {'SimulationMode': True }
    raw_file_name = 'theta_best.dat'
    world_model, sample = init_world_model(Exp, raw_file_name)
    y = np.linspace(0, 20, num=21).astype(np.float32)
    x = np.linspace(-30, 30, num=16).astype(np.float32)
    xx,yy = np.meshgrid(x,y)
    state = np.concatenate((xx[:,:,np.newaxis],yy[:,:,np.newaxis]),2).reshape((-1,2))
    risk_epistemic, risk_aleatoric = [], []

    # predict state by model
    with torch.no_grad():
        # state, action: [Batch *1 *_]
        state = torch.from_numpy(state).to(torch.float32).cuda()
        state = state.unsqueeze(1)
        action = policy(state)
        # reward: [Batch * M * N *1]; state_next:[Batch * M * N *1 *_]
        M, N = 60, 60 # for simulation
        state_next, reward, _, _ = world_model.predict_MC(state, action, M, N)

        # aleatoric, epistemic of state(y-axis): [Batch]
        state_next_y = state_next[:, :, :, 0, 1]
        aleatoric, epistemic = world_model.decompose_uncert(state_next_y, 1., 1.)

        # # aleatoric, epistemic of reward: [Batch]

        risk_epistemic.append(epistemic.cpu().numpy())
        risk_aleatoric.append(aleatoric.cpu().numpy())

    risk_epistemic, risk_aleatoric = np.flipud(np.array(risk_epistemic).squeeze().reshape((len(y),len(x)))),\
                                     np.flipud(np.array(risk_aleatoric).squeeze().reshape((len(y),len(x))))

    dataNew = './Uncert_decom_data/uncert_decom_result.mat'

    scio.savemat(dataNew, {'risk_epistemic': risk_epistemic,
                           'risk_aleatoric': risk_aleatoric,
                           'x': x,
                           'y':y[::-1]
                           })
    print('Done!')
    