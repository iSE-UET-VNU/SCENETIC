import requests
import numpy as np
import pandas as pd
import time
import math
import pickle
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory.buffer import PrioritizedReplayBuffer
from utils import get_environment_state, get_action_space, calculate_distance

from pipeline_constants import *


script_dir = os.path.dirname(os.path.abspath(__file__))

# load model params
continue_episode = ''
reuse_episode = ''
start_eps = '0'
end_eps = '300'

# judge stopping params
check_num = STOP_STEP
position_space = []
position_space_size = 0

road_num = ROAD_NUM  # the Road Number
second = OBSERVATION_TIME # otp
scene = SANFRANCISCO_MAP

request_prefix = 'http://' + API_SERVER_HOST + ':' + str(API_SERVER_PORT) + "/crisis"

requests.post(f"{request_prefix}/load-scene?scene={scene}&road_num={road_num}&saving=0")

file_name = str(int(time.time()))

endpoint_json = open(f'{script_dir}/../configuration_api_server/map_endpoint/ego_endpoint.json', 'r')
endpoint_list = endpoint_json.read()
ego_endpoint = json.loads(s=endpoint_list)

end_point = ego_endpoint[scene]['road' + road_num]['end']

goal = [
    end_point['position']['x'], 
    end_point['position']['y'], 
    end_point['position']['z']
]

action_space = get_action_space()['api']
scenario_description = get_action_space()['description']
N_ACTIONS = action_space['num']

states, _ = get_environment_state()
N_STATES = states.shape[0]

# Log HyperParameters
print(f"Number of action: {N_ACTIONS}")
print(f"Number of state: {N_STATES}")
print(f"HyperParameters: {HyperParameter}")

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, HyperParameter['NUM_LAYER_1'])
        self.fc1.weight.data.normal_(0, HyperParameter['PARAM_INIT_RANGE'])  # initialization
        
        self.fc2 = nn.Linear(HyperParameter['NUM_LAYER_1'], HyperParameter['NUM_LAYER_2'])
        self.fc2.weight.data.normal_(0, HyperParameter['PARAM_INIT_RANGE'])  # initialization
        
        self.out = nn.Linear(HyperParameter['NUM_LAYER_2'], N_ACTIONS)
        self.out.weight.data.normal_(0, HyperParameter['PARAM_INIT_RANGE'])  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DDQN(object):
    def __init__(self):
        # eval and target network
        self.eval_net, self.target_net = Net(), Net()
        
        # learning rate decay
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=HyperParameter['lr'], weight_decay=0)
        
        # buffer memory
        self.steps_done = 0
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.buffer_memory = PrioritizedReplayBuffer(N_STATES, N_ACTIONS, HyperParameter["MEMORY_SIZE"], 
                                                     HyperParameter["PER_EPS"], HyperParameter["PER_ALPHA"], 
                                                     HyperParameter["PER_BETA"])
        
        # actions and probabilities
        self.num_of_npc_action = USE_NPC_VEHICLE * NUMBER_NPC_VEHICLE + USE_PEDESTRIAN * NUMBER_PEDESTRIAN
        self.num_of_action = self.num_of_npc_action + USE_WEATHER_TIME * NUMBER_WEATHER_TIME
        self.action_chosen_prob = [1 / self.num_of_action] * self.num_of_action
        self.npc_action_chosen_prob = [1 / self.num_of_npc_action] * self.num_of_npc_action

    def update_action_prob(self, action, is_testing):
        
        print(20*'-', "Update action probabilities", 20*'-')
        
        prob = self.action_chosen_prob[action]
        
        reduced_amount = prob * PROB_DECAY

        if is_testing:
            reduced_amount = 0
        
        for i in range(0, self.num_of_action):
            if i == action:
                self.action_chosen_prob[i] -= reduced_amount
            else:
                self.action_chosen_prob[i] += reduced_amount / (self.num_of_action - 1)    
                
    def update_npc_action_prob(self, action, is_testing):
        
        print(20*'-', "Update driving action probabilities", 20*'-')
        
        prob = self.npc_action_chosen_prob[action]
        
        reduced_amount = prob * PROB_DECAY

        if is_testing:
            reduced_amount = 0
        
        for i in range(0, self.num_of_npc_action):
            if i == action:
                self.npc_action_chosen_prob[i] -= reduced_amount
            else:
                self.npc_action_chosen_prob[i] += reduced_amount / (self.num_of_npc_action - 1)    

    def choose_action(self, x, no_weather_time_step, prev_weather_time_step, is_testing=False):
        
        print(20*'-', "Choose new action", 20*'-')
        
        choose_method = "by model"
        action = None
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        
        # calculate epsilon
        eps_threshold = HyperParameter['EPS_END'] + (
                HyperParameter['EPS_START'] - HyperParameter['EPS_END']) * math.exp(
            -1. * self.steps_done / HyperParameter['EPS_DECAY'])
                
        if is_testing:
            eps_threshold = 0.2
                
        self.steps_done += 1
        
        # check whether agent learnt enough or not
        if self.steps_done > HyperParameter["EPS_DECAY"]: 
            eps_threshold = 0.1

        print("eps threshold:", eps_threshold)
        
        # decide choosing method
        is_greedy = (np.random.uniform() > eps_threshold)
        if self.steps_done <= HyperParameter["MEMORY_SIZE"] and not is_testing:
            is_greedy = False
            choose_method = 'randomly'
        
        if is_greedy:  # greedy
            print("Choose by model")
            
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
            
            # re-select if action is weather or time
            is_weather_time_action = (USE_WEATHER_TIME and 0 <= action and action <= 12)
            if is_weather_time_action:
                if no_weather_time_step >= 5:
                    if action == prev_weather_time_step:
                        _, topk_indices = torch.topk(actions_value, k=2)
                        action = topk_indices[0][1].data.numpy()
                else:
                    _, topk_indices = torch.topk(actions_value, k=15)
                    cnt = 1
                    while cnt <= 14:
                        action = topk_indices[0][cnt].data.numpy()
                        if not (0 <= action and action <= 12):
                            break
                        
                        cnt += 1
            
        else:  # random
            print("Choose randomly")
           
            action = np.random.choice(self.num_of_action, size=1, replace = False, p=self.action_chosen_prob)
            action = action[0]
            
            # re-select if action is weather or time
            is_weather_time_action = (USE_WEATHER_TIME and 0 <= action and action <= 12)
            if is_weather_time_action:
                if no_weather_time_step >= 5:
                    if action == prev_weather_time_step:
                        action = np.random.choice(self.num_of_action, size=1, replace = False, p=self.action_chosen_prob)
                        action = action[0]
                else:
                    action = np.random.choice(self.num_of_npc_action, size=1, replace = False, p=self.npc_action_chosen_prob) + (self.num_of_action - self.num_of_npc_action)
                    action = action[0]

            # update action's probabilities
            self.update_action_prob(action, is_testing)
            if 12 < action and USE_WEATHER_TIME:
                self.update_npc_action_prob(action - (self.num_of_action - self.num_of_npc_action), is_testing)

        return action, choose_method

    def store_transition(self, s, a, r, s_, done):
        print(20*'-', "Store transition to replay buffer", 20*'-')
        
        self.memory_counter += 1
        self.buffer_memory.add((s, a, r, s_, int(done)))

    def learn(self):
        
        print(20*'-', "Start Training Double Deep-Q Network", 20*'-')
        
        # target parameter update
        if self.learn_step_counter % HyperParameter['TARGET_UPDATE'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        batch, weights, tree_idxs = self.buffer_memory.sample(HyperParameter["BATCH_SIZE"])
        state, action, reward, next_state, _ = batch        
        action = action.max(dim=1).values.unsqueeze(1).to(torch.int64)
        reward = reward.unsqueeze(1)
        
        # calculate q-values for current states and actions
        q_eval = self.eval_net.forward(state).gather(1, action)
        
        # choose next actions for next states
        q_action = self.eval_net.forward(next_state).max(dim=1).indices
        
        # calculate q-values for next states and actions by target network
        q_value = self.target_net.forward(next_state).detach()  # detach from graph, don't backpropagate       
        q_next = q_value.max(dim=1).values
        for i in range(0, HyperParameter['BATCH_SIZE']):
            q_next[i] = q_value[i][q_action[i]]
 
        q_next = q_next.view(-1, 1)        
        q_target = reward + HyperParameter['GAMMA'] * q_next  # shape (batch, 1)
        
        # calculate loss and td_error
        td_error = torch.abs(q_eval - q_target).detach()
        loss = torch.mean(torch.abs(q_eval - q_target) ** 2 * weights)

        pd.DataFrame([[self.learn_step_counter, self.optimizer.param_groups[0]['lr'], loss.item()]]).to_csv(f'{script_dir}/loss_log/loss_log_' + file_name + '.csv', 
                                                         mode='a', header=False, index=None)

        # back propagation and update priorities
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer_memory.update_priorities(tree_idxs, td_error.numpy())
    
def execute_action(action_id):
    api = action_space[str(action_id)]
    print("Start action: ", api)
    
    response = requests.post(api)
    print(response)
    
    obstacle_uid = None
    generated_uid = None
    
    try:
        proc_list = response.json()['probability']
        obstacle_uid = response.json()['collision_uid']
        generated_uid = response.json()['generated_uid']
    except Exception as e:
        print(e)
        proc_list = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        
    return proc_list, obstacle_uid, generated_uid

def judge_done():
    global position_space_size
    global position_space
    
    judge = False
    position = requests.get(f"{request_prefix}/status/ego-vehicle/position").json()
    print("Ego's position: ", position)
    
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
        
        # check general distance
        if dis < STOP_DIS:
            judge = True
            
        # check altitude distance
        if abs(dis2) > STOP_DIS_ALTITUDE:
            judge = True
            
    return judge

def calculate_reward(action_id):

    proc_list, obstacle_uid, generated_uid = execute_action(action_id)
    observation, pos = get_environment_state()
    action_reward = 0
    
    # Collision Probability Reward
    collision_probability = 0
    collision_reward = 0
    
    # get collision info
    collision_info = (requests.get(f"{request_prefix}/status/collision-info")).content.decode(
        encoding='utf-8')
    print("Collision Info: ", collision_info)

    # get collision uid
    col_uid = (requests.get(f"{request_prefix}/status/collision-uid")).content.decode(
        encoding='utf-8')
    print("Collision Uid: ", col_uid)

    episode_done = judge_done()

    # calculate collision reward
    if collision_info != 'None':
        collision_reward = RCOL
        collision_probability = 1.0
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
      
    action_reward = collision_reward
            
    return observation, pos, action_reward, collision_probability, episode_done, proc_list, obstacle_uid, collision_info, col_uid, generated_uid

def check_model_folder():

    model_path = script_dir + "/" + MODEL_PATH

    print("Model folder: ", model_path)
    print("In folder name: ", IN_MODEL_NAME)
    print("Out folder name: ", OUT_MODEL_NAME)
    
    os.makedirs(model_path, exist_ok=True)
    
    in_model_path = model_path + "/" + IN_MODEL_NAME
    out_model_path = model_path + "/" + OUT_MODEL_NAME
    
    if not os.path.isdir(in_model_path):
        print("Create in dir", in_model_path)
        os.makedirs(in_model_path)
    
    if not os.path.isdir(out_model_path):
        print("Create out dir", out_model_path)
        os.makedirs(out_model_path)
        
def check_log_folder():

    log_path = script_dir + "/../" + LOG_PATH
    print("Log path: ", log_path)

    os.makedirs(log_path, exist_ok=True)

def load_model(ddqn, continue_episode):

    model_path = script_dir + "/" + MODEL_PATH
    
    print("Continue at episode: " + continue_episode)
    
    in_model_path = model_path + "/" + IN_MODEL_NAME
        
    with open(f"{in_model_path}/rl_network_{continue_episode}_road{road_num}.pkl", "rb") as file:
        ddqn = pickle.load(file)
    ddqn.eval_net.load_state_dict(torch.load(f"{in_model_path}/eval_net_{continue_episode}_road{road_num}.pt"))
    ddqn.target_net.load_state_dict(torch.load(f"{in_model_path}/target_net_{continue_episode}_road{road_num}.pt"))
    with open(f"{in_model_path}/memory_buffer_{continue_episode}_road{road_num}.pkl", "rb") as file:
        ddqn.buffer_memory = pickle.load(file)       
        
    print(ddqn.buffer_memory.real_size, ddqn.learn_step_counter, ddqn.steps_done)
    
    return ddqn

def load_buffer_memory(ddqn, reuse_episode):

    model_path = script_dir + "/" + MODEL_PATH
    
    print("Reuse memory buffer from episode: " + reuse_episode)
    
    reuse_path = model_path + "/" + REUSE_MEMORY_NAME
        
    with open(f"{reuse_path}/rl_network_{reuse_episode}_road{road_num}.pkl", "rb") as file:
        ddqn = pickle.load(file)
    with open(f"{reuse_path}/memory_buffer_{reuse_episode}_road{road_num}.pkl", "rb") as file:
        ddqn.buffer_memory = pickle.load(file)       
        
    print(ddqn.buffer_memory.real_size, ddqn.learn_step_counter, ddqn.steps_done)
    print(ddqn.action_chosen_prob)
    print(ddqn.npc_action_chosen_prob)
    
    return ddqn

collide_with_obstacle = False
previous_weather_and_time_step = -5
prev_weather_time_step = -1
prev_position = [0, 0, 0]
prev_collision_info = ""
prev_collision_uid = ""
collision_position = [0, 0, 0]
step_after_collision = -1
uid_list = {}

def analysis_collision(curr_pos, obstacle_uid, reward, done):
    
    global prev_collision_uid
    global prev_collision_info
    global collision_position
    global step_after_collision
    global collide_with_obstacle
    global prev_position
    
    # calculate distance to previous collision position
    dis_to_prev_col = calculate_distance(curr_pos, collision_position)       
    if collision_info != 'None':
        is_collision = False
    
        # check whether collision happened or not
        if prev_collision_uid != col_uid:
            is_collision = True
        else:
            if dis_to_prev_col >= 3:
                is_collision = True
            else:
                reward = 0 # if distance didn't change enough, it's not a collision
                
        if is_collision:
            collision_position = [curr_pos[0], curr_pos[1], curr_pos[2]]
            prev_collision_info = collision_info
            prev_collision_uid = col_uid
            step_after_collision = 0
        else:
            # if distance didn't change during 3 step, stop current episode
            if step_after_collision >= 0:
                step_after_collision += 1
                if step_after_collision >= 3:
                    if dis_to_prev_col < 3:
                        done = True
                    else:
                        step_after_collision = -1                                        
    else:
        # if distance didn't change during 3 step, stop current episode
        if step_after_collision >= 0:
            step_after_collision += 1
            if step_after_collision >= 3:
                if dis_to_prev_col < 3:
                    done = True
                else:
                    step_after_collision = -1 
                    
    # Consider whether colliding to obstacle (signal, static obstacle, v.v) or not
    if collide_with_obstacle == True:
        dis_to_prev_step = calculate_distance(prev_position, curr_pos)
        
        if dis_to_prev_step <= 2:
            done = True
    
    if obstacle_uid == 'OBSTACLE':
        collide_with_obstacle = True
    else:
        collide_with_obstacle = False
                    
    return reward, done

def delay_reward(ddqn, generated_uid, collision_info, col_uid, reward):
    
    global uid_list
    
    if generated_uid:
        uid_list[generated_uid] = ddqn.buffer_memory.count
        
    if (collision_info == 'pedestrian' or collision_info == 'npc_vehicle') and col_uid != generated_uid:
        ddqn.buffer_memory.reward[uid_list[col_uid]] = torch.as_tensor(reward)
        reward = reward * 1/3
        
    return ddqn, reward

def judge_done_to_goal(curr_pos, done):
    
    global prev_position
    global goal
    
    dis_to_prev_step = calculate_distance(prev_position, curr_pos)
    dis_to_goal = calculate_distance(curr_pos, goal)
    
    print("Dis ", dis_to_prev_step, "dis to goal " , dis_to_goal)

    if dis_to_prev_step <= 2 and dis_to_goal <= 5:
        done = True
        
    return done

def save_model(i_episode, ddqn):

    model_path = script_dir + "/" + MODEL_PATH
    
    out_model_path = model_path + "/" + OUT_MODEL_NAME
    
    if (i_episode + 1) % 5 == 0:
        
        i_episode += 1
        
        torch.save(ddqn.eval_net.state_dict(), f"{out_model_path}/eval_net_{i_episode}_road{road_num}.pt")
        torch.save(ddqn.target_net.state_dict(), f"{out_model_path}/target_net_{i_episode}_road{road_num}.pt")
        
        with open(f"{out_model_path}/memory_buffer_{i_episode}_road{road_num}.pkl", "wb") as file:
            pickle.dump(ddqn.buffer_memory, file)
            
        with open(f"{out_model_path}/rl_network_{i_episode}_road{road_num}.pkl", "wb") as file:
            pickle.dump(ddqn, file)
            
def restart_episode():
    print("Restart episode")
    
    global collide_with_obstacle
    global prev_position
    global position_space
    global previous_weather_and_time_step
    global prev_collision_info
    global prev_collision_uid
    global collision_position
    global step_after_collision
    global prev_weather_time_step
    
    collide_with_obstacle = False
    prev_position = [0, 0, 0]
    position_space = []
    prev_weather_time_step = -1
    previous_weather_and_time_step = -5
    prev_collision_info = ""
    prev_collision_uid = ""
    collision_position = [0, 0, 0]
    step_after_collision = -1

if __name__ == '__main__':
    
    # init ddqn
    ddqn = DDQN()
    check_model_folder()

    if continue_episode != '':
        ddqn = load_model(ddqn, continue_episode)
        
    if reuse_episode != '':
        ddqn = load_buffer_memory(ddqn, reuse_episode)
        
    print('\nCollecting experience...')
    print("Road num: ", road_num)
    
    # logging
    check_log_folder()
    file_name = str(int(time.time()))
    log_path = script_dir + "/../" + LOG_PATH
    log_name = f"{log_path}/{LOG_NAME}_{file_name}.csv"
    
    title = ["Episode", "Step", "State", "Action", "Reward", "Collision Probability", "Collision Probability List", "Action_Description", "Done"]
    df_title = pd.DataFrame([title])
    df_title.to_csv(log_name, mode='w', header=False, index=None)

    # set up observation time
    requests.post(f"{request_prefix}/set-observation-time?observation_time={second}")
        
    for i_episode in range(int(start_eps), int(end_eps)):
        print('------------------------------------------------------')
        print('+                 Road, Episode: ', road_num, i_episode, '                +')
        print('------------------------------------------------------')
        requests.post(f"{request_prefix}/load-scene?scene={scene}&road_num={road_num}&saving=0")

        s, _ = get_environment_state()
        step = 0
        while True:
            
            # implement action
            action, _ = ddqn.choose_action(s, step - previous_weather_and_time_step, prev_weather_time_step)
            s_, curr_pos, reward, proC, done, proc_list, obstacle_uid, collision_info, col_uid, generated_uid = calculate_reward(action)  
            
            if USE_WEATHER_TIME and 0 <= action and action <= 12:
                previous_weather_and_time_step = step
                prev_weather_time_step = action
            
            # analysis collision if exist
            reward, done = analysis_collision(curr_pos, obstacle_uid, reward, done)
            print(f"Reward after analyzing collision: {reward}")
            
            # delay reward
            ddqn, reward = delay_reward(ddqn, generated_uid, collision_info, col_uid, reward)
            print(f"Reward after delaying: {reward}")
            
            # check if ego is stopping or has reached the goal
            done = judge_done_to_goal(curr_pos, done)
            
            # store transition
            ddqn.store_transition(s, action, reward, s_, done)
            
            if ddqn.memory_counter > HyperParameter['MEMORY_SIZE']:
                ddqn.learn()

            save_model(i_episode, ddqn)
            
            prev_position = [curr_pos[0], curr_pos[1], curr_pos[2]]
            
            step += 1
            
            s = s_
            
            action_description = scenario_description[str(action)]  
            print('>>>>>step, action, reward, collision_probability, action_description, done: ', step, action,
                    reward, round(proC, 6), "<" + action_description + ">", done)
            
            pd.DataFrame(
                [[i_episode, step, s, action, reward, proC, proc_list, action_description, done]]).to_csv(
                log_name,
                mode='a',
                header=False, index=None)

            if done:
                restart_episode()
                break
            