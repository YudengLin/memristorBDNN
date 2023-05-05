# memristorBDNN
###### **Overview**

This code repository is partly to support risk-sensitive reinforcement learning experiment in the submitted manuscript 《Uncertainty quantification via a memristor Bayesian deep neural network for risk-sensitive reinforcement learning》.

###### **System Requirements**

The code requires a standard computer with GPU and enough RAM to support the operations defined by a user.The developmental version of the package has been tested on the `Linux: Ubuntu 16.04` system. And the code mainly depends on the following `Python3.6` scientific packages.

```
pandas 1.1.4
pytorch 1.1.0
tensorflow 1.15.0
tensorboard 
stable-baselines 2.10.2
gym 0.17.2
```

###### **INSTALLATION GUIDE**

Installation not required.

###### **Main Files description**

Since the ESCIM hardware platform is customized designed, the codes that interface the hardware is not provided. The experimental data will be provided upon reasonable request.

1. To train a memristor Bayesian deep neural network (BDNN) to obtain target conductance, run:
   >python train_world_model.py 

   Output:
    ```
    Data: ./Dataset/out/2D/
    
    Network:
    [5, 100, 100, 2]
    [5, 100, 100, 2]
    Start epochs!
    Epoch: 001/3000, Train loss = 17400.498 z_rhos = 5.250 noise_rhos = -2.980
    Epoch: 002/3000, Train loss = 18035.643 z_rhos = 5.249 noise_rhos = -2.961
    Epoch: 003/3000, Train loss = 15602.429 z_rhos = 5.248 noise_rhos = -2.942
    ......
    ```

    The trained model will be saved in path `./BDNN_models/theta_best.dat`
2. To quantify two prediction uncertainties using memristor BDNN, run:
    >python uncertainty_decomposition.py
   
    The uncertainty results will be saved in `./Uncert_decom_data/uncert_decom_result.mat`
3. To train a policy through PPO algorithm, run:
   >cd PPO
   >
   >python train_PPO.py

    The policy model will be saved in `./PPO/logs/b40g0.95/best_model.zip`

###### **License**

This code repository is covered under the Apache 2.0 License.
