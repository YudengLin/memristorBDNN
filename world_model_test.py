import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import util as tools
import torch
from Dataset.mydataset import Dataset
from BDNN_Model import BBP_Model
from BDNN_Model_Wrapper import BBP_Model_Wrapper
import numpy as np
from Dataset.Environment import Environment

def init_world_model(Exp, file_name):
    # %% dataset
    dirs = './Dataset/out/2D/'
    print('\nData: ' + dirs)
    preprocess_data = False
    trainset = Dataset(root=dirs, preprocess_data=preprocess_data)

    batch_size = 200
    NTrainPoints = trainset.len()
    nb_train = int(NTrainPoints / batch_size)
    output_dim = 2
    input_dim = 4
    layer_size = [input_dim, 100, 100, output_dim]
    learn_rate = 0.01
    batch_size = 100
    sample = 360
    model = BBP_Model(layer_size, NTrainPoints, sample, Exp).cuda()
    world_model = BBP_Model_Wrapper(model, learn_rate, trainset, batch_size, nb_train)

    PATH = './BDNN_models/'
    print('\nWorld model:' + file_name)
    if Exp['SimulationMode']:
        model.load_state_dict(torch.load(PATH+file_name))
    else:
        # using plat, and  will load quant scale paras
        print('Change to use platform, and quant input mode')
        model.load_state_dict(torch.load(PATH+file_name),strict=False)
    model.eval()

    return world_model, sample

def plot_dist(world_model,state, action, no_sample):
    # init true environment
    Env = Environment()
    # true state by Environment
    environ_state = []
    for s in range(no_sample):
        Env.set(state,clear_state_trans=True)
        s_next, r, dead,_ = Env.step(action)
        if not dead:
            environ_state.append(s_next)
    groundtrue = np.array(environ_state)
    groundtrue_y = groundtrue[:, 1]

    # predict state by model
    with torch.no_grad():
        state, action = torch.from_numpy(state).to(torch.float32).cuda(), torch.from_numpy(action).to(torch.float32).cuda()
        state = state.unsqueeze(0).unsqueeze(0).expand(no_sample, -1, -1)
        action = action.unsqueeze(0).unsqueeze(0).expand(no_sample, -1, -1)
        input_z = (torch.randn((no_sample, 1, 1))*torch.sqrt(world_model.network.v_prior_z)).cuda()
        output_noise = torch.randn((no_sample, 1, world_model.network.output_dim)).cuda()
        ext_noises = [output_noise, input_z]
        state_t1,  reward_t1, game_over_t1, cost, ave_fly_time = world_model.predict(state, action, no_sample,
                                                                                     ext_noises)
        predict_state = state_t1.cpu().numpy()
        # print('ave_fly_time:'+str(ave_fly_time))

    predict_y = np.array(predict_state)[:, 0, 1]
    predict_y = predict_y[predict_y>0.]

    bins = 20
    # calc JS divergence
    js_div = tools.compute_js_divergence(groundtrue_y, predict_y,bins)

    return  js_div

def calc_state_JS(world_model, sample, Exp, plot = False):

    y = np.arange(1, 11, 2)
    x = np.arange(-10, 22, 10)

    xx, yy = np.meshgrid(x, y)
    js_total = []
    for sx, sy in zip(np.reshape(xx, (1, -1)).squeeze(), np.reshape(yy, (1, -1)).squeeze()):
        st = np.array([sx, sy])
        at = np.array([0., 0.])
        js_div = plot_dist(world_model, st, at, sample,Exp, plot=plot)
        js_total.append(js_div)

    return js_total
