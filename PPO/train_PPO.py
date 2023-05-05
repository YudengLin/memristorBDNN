from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1 as RL
from Env import make_env
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if __name__ == '__main__':
    # SimulationMode:
    #   True    - use memristor simulation model
    #   False   - use ESCIM computing platform
    # Env:
    #   0: use true environment
    #   1: use world model with uncertainty decomposition results
    Exp = {'SimulationMode': True,
           'Env': 1
           }
    raw_file_name = 'theta_best.dat'
    risk_beta, risk_gamma = 40., 0.95
    print('risk_beta, risk_gamma = {},{}'.format(risk_beta, risk_gamma))
    log_path = './logs/b40g0.95'
    train_env = make_env(Exp, raw_file_name, risk_beta, risk_gamma)
    eval_env = make_env(Exp={'Env': 0},file_name=raw_file_name)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                 log_path=log_path,
                                 n_eval_episodes=100,
                                 eval_freq=5000,
                                 deterministic=True, render=False)
    model = RL(MlpPolicy, train_env, gamma=0.99,
                timesteps_per_actorbatch=512,
                clip_param=0.2,
                entcoeff=0.03,
                verbose=1,tensorboard_log="./tensorboard/b40g0.95/")
    model.learn(total_timesteps=50001, callback=[eval_callback])
    if Exp['Env'] != 0:
        del train_env.model
        del train_env.model
        del train_env

