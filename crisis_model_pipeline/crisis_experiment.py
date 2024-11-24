from utils import get_action_space, get_environment_state

import requests
import pandas as pd
import time
import numpy as np
import torch 
from crisis_training_model import DDQN
import json
import os

from pipeline_constants import *

script_dir = os.path.dirname(os.path.abspath(__file__))

request_prefix = 'http://' + API_SERVER_HOST + ':' + str(API_SERVER_PORT) + "/crisis"

scene = SANFRANCISCO_MAP
road_num = ROAD_NUM
second = OBSERVATION_TIME
test_model_eps = TEST_MODEL_EPS

# Execute action
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    response = requests.post(api)
    print(response)
    
    obstacle_uid = None
    sudden_appearance = False
    overlapping = False
    position_list = []
    is_collision_ahead = False
    pedes_mov_fw_to = False
    
    try:
        proc_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        sudden_appearance = response.json()['sudden_appearance']
        overlapping = response.json()['overlapping']
        position_list = response.json()['position_list']
        is_collision_ahead = response.json()['isCollisionAhead']
        pedes_mov_fw_to = response.json()['pedes_mov_fw_to']
    except Exception as e:
        print(e)
        proc_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proc_list, obstacle_uid, sudden_appearance, overlapping, position_list, is_collision_ahead, pedes_mov_fw_to


check_num = STOP_STEP
position_space = []
position_space_size = 0

def judge_done():
    global position_space_size
    global position_space
    
    judge = False
    position = requests.get(f"{request_prefix}/status/ego-vehicle/position").json()
    position_space.append((position['x'], position['y'], position['z']))
    position_space_size = (position_space_size + 1) % check_num
    if len(position_space) == check_num:
        start_pos = position_space[0]
        end_pos = position_space[check_num - 1]
        position_space = []
        dis = pow(
            pow(start_pos[0] - end_pos[0], 2) + pow(start_pos[1] - end_pos[1], 2) + pow(start_pos[2] - end_pos[2], 2),
            0.5)
        
        dis2 = start_pos[1] - end_pos[1]
        
        if dis < STOP_DIS:
            judge = True
            
        if abs(dis2) > STOP_DIS_ALTITUDE:
            judge = True
    return judge

# Execute action and get return
def calculate_reward(action_id):
    
    proc_list, obstacle_uid, sudden_appearance, overlapping, position_list, is_collision_ahead, pedes_mov_fw_to = execute_action(action_id)
    observation, pos = get_environment_state()

    # Collision Probability Reward
    
    collision_probability = 0
    collision_reward = 0
    
    collision_info = (requests.get(f"{request_prefix}/status/collision-info")).content.decode(
        encoding='utf-8')

    print("Collision Info: ", collision_info)

    episode_done = judge_done()

    if collision_info != 'None':
        collision_reward = RCOL
        collision_probability = 1
    elif collision_info == "None":
        collision_probability = round(float(
            (requests.get(f"{request_prefix}/status/collision-probability")).content.decode(
                encoding='utf-8')), 6)
        if collision_probability < PENALTY_THRESHOLD:
            collision_reward = PENALTY
        elif PENALTY_THRESHOLD <= collision_probability < 1.0:
            collision_reward = collision_probability
        else:
            collision_reward = RCOL
        
    print("Collision Probability: ", collision_probability)
    print("Collision Reward: ", collision_reward)
            
    return observation, collision_probability, episode_done, proc_list, obstacle_uid, collision_info, sudden_appearance, overlapping, position_list, is_collision_ahead, pedes_mov_fw_to

def check_test_folder():

    test_prefix_path = script_dir + "/../" + TEST_PATH
    
    test_path = f"{test_prefix_path}/{TEST_NAME}/"
    
    os.makedirs(test_path, exist_ok=True)
    
    return test_path
    
def check_model_path():

    model_prefix_path = script_dir + "/" + MODEL_PATH
    
    model_path = f"{model_prefix_path}/{TEST_NAME}"
    
    if not os.path.isdir(model_path):
        raise Exception("Model path not found")
    
    return model_path

def unreal_case_log(sudden_appearance, overlapping, repeated_collision, not_collision_forward, pedes_mov_fw_to):
    if sudden_appearance:
        print(20*'*', "Scenario is removed due to sudden appearance", 20*'*')
        
    if overlapping:
        print(20*'*', "Scenario is removed due to overlapping", 20*'*')
        
    if repeated_collision:
        print(20*'*', "Scenario is removed due to collision is repeated", 20*'*')
        
    if not_collision_forward:
        print(20*'*', "Scenario is removed due to pedestrian did not collide ahead", 20*'*')
        
    if pedes_mov_fw_to:
        print(20*'*', "Scenario is removed due to pedestrian moves forward to ego", 20*'*')

previous_weather_and_time_step = -5
prev_weather_time_step = -1
npc_interaction_info = {}
state_ = []
action_ = []
type_ = []
probability_ = []
collision_uid_ = []
sudden_appearance_ = []
overlapping_ = []
collision_probability_per_step_ = []
done_ = []
is_collision = False
position_list_ = []
repeated_collision_ = []
unreal_pedes_col_ = []

def check_repeated_collision(position_list, collision_info, done):
    
    global npc_interaction_info
    
    repeated_collision = False
    no_check_repeated = False
    
    if position_list:
        for time_step in position_list.values():
            for npc_uid in time_step:
                if npc_uid == collision_uid:
                    if npc_uid in npc_interaction_info:
                        
                        if collision_info == 'pedestrian' and not no_check_repeated:
                            done = True
                            repeated_collision = True
                        
                        if npc_interaction_info[npc_uid]['collision'] and not no_check_repeated:
                            repeated_collision = True
                        else:
                            npc_interaction_info[npc_uid]['collision_distance'] = time_step[npc_uid]['dis_to_ego']
                            npc_interaction_info[npc_uid]['collision'] = True 
                            no_check_repeated = True
                    else:
                        npc_interaction_info[npc_uid] = {}
                        npc_interaction_info[npc_uid]['collision'] = True
                        npc_interaction_info[npc_uid]['collision_distance'] = time_step[npc_uid]['dis_to_ego']
                        no_check_repeated = True
                else:
                    if npc_uid in npc_interaction_info:
                        if abs(time_step[npc_uid]['dis_to_ego'] - npc_interaction_info[npc_uid]['collision_distance']) > 2:
                            npc_interaction_info[npc_uid]['collision'] = False
                            
    return repeated_collision, done


def check_unreal_pedes_col(probability, collision_info, is_collision_ahead, pedes_mov_fw_to):
    unreal_pedes_col = False
    not_collision_forward = False
    
    if probability == 1.0 and collision_info == 'pedestrian':
        if not is_collision_ahead:
            unreal_pedes_col = True
            not_collision_forward = True
        else:
            if pedes_mov_fw_to:
                unreal_pedes_col = True
                
    return unreal_pedes_col, not_collision_forward

def restart_episode():
    
    global state_
    global action_
    global type_
    global probability_
    global collision_uid_
    global sudden_appearance_
    global overlapping_
    global collision_probability_per_step_
    global done_
    global is_collision
    global position_list_
    global repeated_collision_
    global unreal_pedes_col_
    global previous_weather_and_time_step
    global prev_weather_time_step
    global npc_interaction_info
    
    state_ = []
    action_ = []
    type_ = []
    probability_ = []
    collision_uid_ = []
    sudden_appearance_ = []
    overlapping_ = []
    collision_probability_per_step_ = []
    done_ = []
    is_collision = False
    position_list_ = []
    repeated_collision_ = []
    unreal_pedes_col_ = []
    previous_weather_and_time_step = -5
    prev_weather_time_step = -1
    npc_interaction_info = {}
    
def log_action_info(iteration, test_log_path):
    global state_
    global action_
    global type_
    global probability_
    global collision_uid_
    global sudden_appearance_
    global overlapping_
    global collision_probability_per_step_
    global done_
    global is_collision
    global position_list_
    global repeated_collision_
    global unreal_pedes_col_
    
    print("Episode complete")
    
    for iter in range(0, len(state_)):
        pd.DataFrame([[iteration, state_[iter], action_[iter], type_[iter], probability_[iter], collision_uid_[iter], 
                        collision_probability_per_step_[iter], sudden_appearance_[iter], overlapping_[iter], repeated_collision_[iter], unreal_pedes_col_[iter], done_[iter]]]).to_csv(
            test_log_path,
            mode='a',
            header=False, index=None)
                
if __name__ == '__main__':
    for loop in range(0, TEST_ATTEMPT):
        requests.post(f"{request_prefix}/load-scene?scene={scene}&road_num={road_num}&saving=1")
        requests.post(f"{request_prefix}/set-observation-time?observation_time={second}")

        action_space = get_action_space()['api']
        action_space_size = action_space['num']
        print("Number of actions: ", action_space_size)

        file_name = str(int(time.time()))

        test_path = check_test_folder()
        model_path = check_model_path()

        # load model
        ddqn = DDQN()
        ddqn.eval_net.load_state_dict(torch.load(f"{model_path}/eval_net_{test_model_eps}_road{road_num}.pt"))
        ddqn.target_net.load_state_dict(torch.load(f"{model_path}/target_net_{test_model_eps}_road{road_num}.pt"))

        # logging
        title = ["Episode", "State", "Action", "Choosing_Type", "Collision_Probability",
                "Collision_uid", "Collision_Probability_Per_Step", "Sudden Appearance", 
                "Overlapping", "Repeated Collision", "Unreal Pedes Col", "Done"]

        df_title = pd.DataFrame([title])
        test_log_path = f"{test_path}/crisis_experiment_{test_model_eps}eps_{file_name}.csv"
        df_title.to_csv(test_log_path, mode='w', header=False, index=None)

        iteration = 0
        step = 0
        print("Start episode 0")

        s, _ = get_environment_state()
        restart_episode()

        while True:
                
            retry = True
            
            while(retry):
            
                retry = False
            
                try:
                    action, type = ddqn.choose_action(s, step - previous_weather_and_time_step, prev_weather_time_step, True)
                    s_, probability, done, collision_probability_per_step, collision_uid, collision_info, sudden_appearance, overlapping, position_list, is_collision_ahead, pedes_mov_fw_to = calculate_reward(action)
                    
                    repeated_collision, done = check_repeated_collision(position_list, collision_info, done)    
                    unreal_pedes_col, not_collision_forward = check_unreal_pedes_col(probability, collision_info, is_collision_ahead, pedes_mov_fw_to)
                except json.JSONDecodeError as e:
                    print(e)    
                    retry = True 

            if 0 <= action and action <= 12:
                previous_weather_and_time_step = step
                prev_weather_time_step = action
                
            if overlapping:
                done = True
            
            if probability == 1.0:
                is_collision = True
            
            state_.append(s)
            action_.append(action)
            type_.append(type)
            probability_.append(probability)
            collision_uid_.append(collision_uid)
            sudden_appearance_.append(sudden_appearance)
            overlapping_.append(overlapping)
            collision_probability_per_step_.append(collision_probability_per_step)
            done_.append(done)
            position_list_.append(position_list) 
            repeated_collision_.append(repeated_collision)
            unreal_pedes_col_.append(unreal_pedes_col)
            
            unreal_case_log(sudden_appearance, overlapping, repeated_collision, not_collision_forward, pedes_mov_fw_to)
            
            print('api_id, probability, sudden_appearance, overlapping, repeated_collision, unreal_pedes_col, done: ', action, probability, sudden_appearance, overlapping, repeated_collision, unreal_pedes_col, done)

            step += 1
            s = s_
            if done:
                is_saving = 1

                print("Length of episode: ", len(state_))
                if len(state_) <= check_num:
                    if not is_collision:
                        print("Restart this episode")
                        is_saving = 0
                    else:
                        log_action_info(iteration, test_log_path)
                        iteration += 1
                else:
                    log_action_info(iteration, test_log_path)
                    iteration += 1
                    
                restart_episode()
                
                requests.post(f"{request_prefix}/load-scene?scene={scene}&road_num={road_num}&saving={is_saving}")
            
                print("Start episode ", iteration)
                if iteration == TEST_SCENARIO_NUM + 1:
                    break
