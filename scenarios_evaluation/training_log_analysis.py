import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def calculate_accumulated_reward(reward_list):
    
    accumulated_reward = 0
    
    gamma = 0.9
    
    for reward in reward_list[::-1]:
        accumulated_reward = reward + gamma * accumulated_reward
        
    return accumulated_reward

def calculate_total_reward(reward_list):
    return sum(reward_list)

def training_analysis(log_file):
    
    df = pd.read_csv(log_file)
    
    current_eps = -1
    cnt_eps = 0
    eps_interval = 15
    
    accumulated_reward_list = []
    total_reward_list = []
    reward_list = []

    collision_scenario = 0
    is_collision = False
    
    avg_return = 0
    avg_return_list = []
    
    num_step = 0
    
    for _, row in df.iterrows():
        
        num_step += 1
        
        eps = row['Episode']
        reward = row['Reward']
        
        if current_eps != eps:
            
            cnt_eps += 1
            current_eps = eps
            
            if eps > 0:
                accumulated_reward = calculate_accumulated_reward(reward_list)
                total_reward = calculate_total_reward(reward_list)
                
                accumulated_reward_list.append(accumulated_reward)
                total_reward_list.append(total_reward)
                
                if is_collision:
                    collision_scenario += 1
                    
                avg_return += accumulated_reward
                if cnt_eps >= eps_interval:
                    avg_return -= accumulated_reward_list[cnt_eps - eps_interval]
                    avg_return_list.append(avg_return / eps_interval)
            
            reward_list = []
            is_collision = False
            
        if float(reward) >= 1.0: 
            is_collision = True
            
        reward_list.append(reward)
    
    accumulated_reward = calculate_accumulated_reward(reward_list)
    total_reward = calculate_total_reward(reward_list)
                
    accumulated_reward_list.append(accumulated_reward)
    total_reward_list.append(total_reward)
    
    avg_return -= accumulated_reward_list[cnt_eps - eps_interval]
    avg_return_list.append(avg_return / eps_interval)
    
    print(collision_scenario)        
    print("Number of step: ", num_step)  
            
    plt.plot(avg_return_list)
    plt.title("Average return in last " + str(eps_interval) + " episode")
    
    plt.xlabel("Episode")
    plt.ylabel("Avg Return")
    
    plt.show()    
 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("log_file", help="Training Log File")

    args = parser.parse_args()

    training_analysis(args.log_file)