import json

import requests
import numpy as np
import math
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os

from pipeline_constants import *

script_dir = os.path.dirname(os.path.abspath(__file__))

request_prefix = 'http://' + API_SERVER_HOST + ':' + str(API_SERVER_PORT) + "/crisis"

# if gpu is to be used
device = torch.device("cpu")

def calculate_distance(src, dest):
    return math.sqrt((src[0] - dest[0]) ** 2 + (src[1] - dest[1]) ** 2 + (src[2] - dest[2]) ** 2)

def get_action_space():
    json_file = open(f'{script_dir}/../restful_api/api_service_list.json', 'r')
    content = json_file.read()
    restful_api = json.loads(s=content)
    return restful_api

def get_environment_state():
    print(20*'-', "Get environment states", 20*'-')
    
    r = requests.get(f"{request_prefix}/status/environment/state")
    a = r.json()
    state = np.zeros(23)
    state[0] = a['lane_info']
    state[1] = a['junction_info']   # check if exist left / right lane or ego is on a junction
    state[2] = a['weather']
    state[3] = a['timeofday']
    state[4] = a['signal']
    state[5] = a['rx']
    state[6] = a['ry']
    state[7] = a['rz']
    state[8] = a['speed']
    
    # add advanced external states 
    state[9] = a['num_obs']
    state[10] = a['min_obs_dist']
    state[11] = a['speed_min_obs_dist']
    
    # add localization option
    
    state[12] = a['local_diff']
    state[13] = a['local_angle']
    
    # add perception option
    state[14] = a['dis_diff']
    state[15] = a['theta_diff']
    state[16] = a['vel_diff']
    state[17] = a['size_diff']
    
    # add control option
    state[18] = a['throttle']
    state[19] = a['brake']
    state[20] = a['steering_rate']
    state[21] = a['steering_target']
    state[22] = a['acceleration']

    pos = np.zeros(3)
    pos[0] = a['x']
    pos[1] = a['y']
    pos[2] = a['z']

    return state, pos


if __name__ == '__main__':
    get_action_space()
