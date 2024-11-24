import numpy as np
import pandas as pd
import argparse

from .evaluation_constants import *

obs_time = OBSERVATION_TIME

def metrics_extract(exp_file):
    
    normal_collision = 0
    distinct_collision = 0
    potential_collision = 0
    
    sudden_appearance_ = 0
    repeated_collision_ = 0
    overlapping_ = 0
    unreal_pedes_col_ = 0
    
    num_realistic_collision = 0
    time_to_collision = 0
    avg_ttc = 0
    
    proc_list = [0]
    avg_proc = 0
    
    is_unrealistic = False
    is_collision = False
    
    prev_position = np.array([0, 0, 0])
    prev_uid = ""
    pre_eps = -1
    
    df = pd.read_csv(exp_file)
    
    for index, row in df.iterrows():
        
        # extract information from scenarios
        eps = row['Episode']
        
        state = row['State'][1:-1].split(",")[0].split(" ")
        state_ = [s for s in state if s != ""]
        
        probability = float(row['Collision_Probability'])
        cp_list = row['Collision_Probability_Per_Step']
        
        uid = row["Collision_uid"]
        
        done = row['Done']
        
        sudden_appearance = row['Sudden Appearance']
        overlapping = row['Overlapping']
        repeated_collision = row['Repeated Collision']
        unreal_pedes_col = row["Unreal Pedes Col"]
        
        if eps != pre_eps:
            # calculate average collision probability in one step of a scenario
            avg_proc_per_eps = sum(proc_list) / max(len(proc_list), 1)
            avg_proc += avg_proc_per_eps
            
            proc_list = []
            is_unrealistic = False
            pre_eps = eps

        if probability > 0: 
            if probability == 1.0:
                   
                normal_collision += 1
            
                if not overlapping and not sudden_appearance and not repeated_collision and not unreal_pedes_col:
                    if prev_uid == uid:
                        new_pos = np.array([float(state_[0]), float(state_[1]), float(state_[2])])
                        diff = np.linalg.norm(prev_position - new_pos)
                        
                        # if ego collided to a vehicle many times but not changed in position enough, it's not a meaningful collision
                        if diff > REPEATED_DISTANCE_THRESHOLD:
                            distinct_collision += 1
                            if not is_unrealistic:
                                proc_list.append(probability)
                        else:
                            repeated_collision_ += 1
                            is_unrealistic = True
                            
                    else:
                        distinct_collision += 1
                        if not is_unrealistic:
                            proc_list.append(probability)
                        
                    prev_position = np.array([float(state_[0]), float(state_[1]), float(state_[2])])
                    prev_uid = uid
    
                    # calculate time to the first collision
                    if not is_collision:
                        max_val = 0.0
                        index = 0
                        cnt = 0
                        
                        for cp in cp_list[1:-1].split(","):
                            fcp = float(cp)
                            if fcp > max_val:
                                max_val = fcp
                                index = cnt
                                
                            cnt += 1
                            
                        time_to_collision += 0.5 * (index + 1)
                    
                    is_collision = True
                else:
                    if sudden_appearance:
                        sudden_appearance_ += 1
                        
                    if repeated_collision:
                        repeated_collision_ += 1
                
                    if overlapping:
                        overlapping_ += 1
                        
                    if unreal_pedes_col:
                        unreal_pedes_col_ += 1
                        
                    is_unrealistic = True
                        
            else:
                # calculate time to the first collision
                if probability > POTENTIAL_THRESHOLD:
                    potential_collision += 1
                    
                if not is_collision:
                    time_to_collision += obs_time
                 
                if not is_unrealistic:
                    proc_list.append(probability)
                
        else:
            # calculate time to the first collision
            if not is_collision:
                time_to_collision += obs_time
            if not is_unrealistic:
                proc_list.append(probability)

                
        if done:
            num_realistic_collision += int(is_collision)
            avg_ttc += time_to_collision * int(is_collision)
            
            is_collision = False

            time_to_collision = 0
            prev_position = np.array([0, 0, 0])
            prev_uid = ""
        
    avg_proc_per_eps = sum(proc_list) / max(len(proc_list), 1)
    avg_proc += avg_proc_per_eps
    avg_proc /= TEST_SCENARIO_NUM
                
    print("Collision: ", normal_collision)
    print("Potential Collision: ", potential_collision)
    print("Distinct Collision: ", distinct_collision) 
    print("Average time to collision: ", avg_ttc / max(num_realistic_collision, 1))
    print("Sudden Appearance: ", sudden_appearance_)
    print("Overlapping: ", overlapping_)
    print("Repeated Collision: ", repeated_collision_)
    print("Unreal Pedes Col: ", unreal_pedes_col_)
    print("Average Collision Probability: ", avg_proc)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")

    args = parser.parse_args()

    metrics_extract(args.exp_file)