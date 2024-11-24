import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import json

obs_time = 6.0

def action_analysis(exp_file, n_actions):
    
    df = pd.read_csv(exp_file)
    n_actions_ = int(n_actions)
    
    action_by_model = np.zeros((n_actions_,))
    action_randomly = np.zeros((n_actions_,))
    
    json_file = open('../RESTfulAPIProcess/RESTful_API_fine_grained.json', 'r')
    content = json_file.read()
    restful_api = json.loads(s=content)["api"]
    
    for _, row in df.iterrows():
        action = row['Action']
        type = row["Choosing_Type"]
        if type == "by model":
            action_by_model[int(action)] += 1
        else:
            action_randomly[int(action)] += 1
                
    print("Action chosen by model: ", action_by_model)
    print("Action chosen randomly: ", action_randomly)
    
    sum = 0
    
    weather = 0
    time = 0
    pedes = 0
    drive_ahead = 0
    overtake = 0
    drive_opposite = 0
    cross_road = 0
    
    for i in range(0, len(action_by_model)):
        sum += action_by_model[i]
        
        if "Weather" in restful_api[str(i)]:
            weather += action_by_model[i]
        elif "TimeOfDay" in restful_api[str(i)]:   
            time += action_by_model[i]
        elif "pedestrian" in restful_api[str(i)]:
            pedes += action_by_model[i]
        elif "drive-ahead" in restful_api[str(i)]:
            drive_ahead += action_by_model[i]
        elif "drive-opposite" in restful_api[str(i)]:
            drive_opposite += action_by_model[i]
        elif "overtake" in restful_api[str(i)]:
            overtake += action_by_model[i]
        elif "cross-road" in restful_api[str(i)]:
            cross_road += action_by_model[i]

    print(sum)
    
    print("Weather: ", weather / sum * 100, "%")
    print("Time: ", time / sum * 100, "%")
    print("Pedestrian: ", pedes / sum * 100, "%")
    print("Drive-Ahead: ", drive_ahead / sum * 100, "%")
    print("Drive-Opposite: ", drive_opposite / sum * 100, "%")
    print("Overtake: ", overtake / sum * 100, "%")
    print("Cross-Road: ", cross_road / sum * 100, "%")
    
    plt.plot(action_by_model)
    plt.xlabel('Action')
    plt.ylabel('Freq')
    plt.title('Action x Freq')
    plt.show()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("exp_file", help="Experiment File")
    parser.add_argument("n_actions", help="Number of actions")

    args = parser.parse_args()

    action_analysis(args.exp_file, args.n_actions)