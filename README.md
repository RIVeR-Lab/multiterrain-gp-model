
## Skid-Steer Robot Modeling

### Introduction
This repository contains the code associated to our ICRA 2024 paper : A Probabilistic Motion Model for Skid-Steer Wheeled Mobile Robot Navigation on Off-Road Terrains.

Paper : https://arxiv.org/pdf/2402.18065.pdf

Video : https://www.youtube.com/watch?v=_rVy2aBp42c

We train Gaussian Process Regression models to predict future robot linear and angular velocity states for different terrains. The outputs of multiple models
are then fused online using a convex optimization formulation allowing the motion model to generalize to different/unseen terrain conditions. The resultant mean and
covariance estimates of the robot states can be used for Risk-Aware Motion Planning approaches such as Stochastic Model Predictive Control. 

### Experimental Setup
Begin by cloning this repository and setting up a Python virtual environment.

```
git clone git@github.com:RIVeR-Lab/multiterrain-gp-model.git
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirments.txt
```

In order to evaluate and benchmark our proposed modeling method, we used the off-road navigation dataset released as a part of [this](https://ieeexplore.ieee.org/abstract/document/8794216) paper. 
Begin by cloning this repository and downloading the dataset as follows. If you have troubles setting up the dataset as suggested above, you can manually download it from [this](https://drive.google.com/file/d/10YAQsaLhTnNbBER5beItwMlTBYkLmqTC/view?usp=drive_link) link. 

```
sudo apt-get install unzip
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
@inproceedings{trivedi2024probabilistic,
  title={A probabilistic motion model for skid-steer wheeled mobile robot navigation on off-road terrains},
  author={Trivedi, Ananya and Zolotas, Mark and Abbas, Adeeb and Prajapati, Sarvesh and Bazzi, Salah and Pad{\i}r, Ta{\c{s}}kin},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={12599--12605},
  year={2024},
  organization={IEEE}
}
```
