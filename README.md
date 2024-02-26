
## Skid-Steer Robot Modeling

### Introduction
This repository contains the code associated to our paper : A Probabilistic Motion Model for Skid-Steer Wheeled Mobile Robot Navigation on Off-Road Terrains.

Paper : (TODO: Add link to arxiv)

Video : (TODO: Add link to RIVeR youtube)

We train Gaussian Process Regression models to predict future robot linear and angular velocity states for different terrains. The outputs of multiple models
are then fused online using a convex optimization formulation allowing the motion model to generalize to different/unseen terrain conditions. The resultant mean and
covariance estimates of the robot states can be used for Risk-Aware Motion Planning approaches such as Stochastic Model Predictive Control. 

### Experimental Setup
Begin by cloning this repository and setting up a Python virtual environment.

```
git clone git@github.com:trivediana/jackal_robot_modeling.git (TODO: add the right link)
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirments.txt
```

In order to evaluate and benchmark our proposed modeling method, we used the off-road navigation dataset released as a part of [this](https://ieeexplore.ieee.org/abstract/document/8794216) paper. 
Begin by cloning this repository and downloading the dataset as follows. If you have troubles setting up the dataset as suggested above, you can manually download it from [this](https://drive.google.com/file/d/10YAQsaLhTnNbBER5beItwMlTBYkLmqTC/view?usp=drive_link) link. 

```
sudo apt-get install unzip
cd jackal_robot_modeling/
gdown 10YAQsaLhTnNbBER5beItwMlTBYkLmqTC
unzip -qq data.zip
rm -rf data.zip
```

The training of the GP/Benchmark kinematic models and their subsequent inference on a test dataset has been assembled into a single script shown below. The plots and tables in the paper were generated via running the individual components of this script.

```
python3 src/probabilistic_dynamics.py
```

### Citation
If you find this code useful, please consider citing
```
TODO: Add link here
```