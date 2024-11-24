import json
import os
import time

from openai import OpenAI


class LLMAPI(object):

    def __init__(self):
        self.total_time_step = 6

    def openai_gpt_4(self, model, messages, temperature):
        
        api_key_path = "E:/Exercise/NCKH/openai-api-key.txt"
        
        api_key = ""
        
        with open(api_key_path, 'r') as file:
            api_key = file.read()

        client = OpenAI(api_key=api_key)  # please add your api_key here

        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=temperature,
        )
        create_time = time.time() - start_time
        middle_time = time.time()
        output = completion.choices[0].message.content
        output_time = time.time() - middle_time
        total_time = time.time() - start_time
        return output, create_time, output_time, total_time


    def weather_condition(value):
        if value == 0:
            return "No"
        elif value == 0.2:
            return "Light"
        elif value == 0.5:
            return "Moderate"
        elif value == 1:
            return "Heavy"
        else:
            return "Unknown"

    def evaluate_R_MR_extra_4full_experiments(self, model, repeat_time, attempt):
        MAX_CHUNK_SIZE = 110000
        road_list = ["road1", "road2", "road3", "road4", "road33"]
        # road_list = ["road33"]

        scenarios_description = "In our experiment, a scenario is created by incrementally introducing dynamic obstacles, such as vehicles and pedestrians, throughout the driving duration."

        road_description = {
            "road1": scenarios_description + "In the following scenario, Ego's driving intention is to first drive on a dual-way road with four lanes, then turn right to a one-way road of four lanes. At the next junction, Ego intends to turn left onto a one-way road, drive a short distance, and then stop.\n",
            "road2": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a four-lane, two-way road, then turn right at a T-intersection. After a short distance, Ego will turn left onto a one-way road.\n",
            "road3": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a six-lane, two-way road, pass one intersection, and then turn left at the second intersection.\n",
            "road4": scenarios_description + "In the following scenario, Ego's driving intention is to first drive on a straight, two-lane road, pass one intersection, turn left, and then make another left at the end of the road, where it connects to a curved boulevard.\n",
            "road33": scenarios_description + "In the following scenario, Ego's driving intention is to first travel on a four-lane, two-way road, then turn right at a T-intersection. After a short distance, Ego will turn left onto a one-way road.\n",}
   
        for road in road_list:
            # Specify the path to the scenario JSON file
            randomly_select_scenarios_path = "" + road + "-scenarios/"
            scenarios_names_list = [f for f in os.listdir(randomly_select_scenarios_path) if
                                                    os.path.isfile(os.path.join(randomly_select_scenarios_path, f))]


            for j in range(11,12):
                i = 16 * (attempt-1) + j
                scenario_name = f"scenario{i}.json"
                collided = False
                randomly_select_scenario_path = randomly_select_scenarios_path + scenario_name
                print(randomly_select_scenario_path)
                with open(randomly_select_scenario_path, 'r') as json_file:
                    randomly_select_scenario = json.load(json_file)
                self.total_time_step = len(randomly_select_scenario.keys())-1
                print(self.total_time_step)
                object_refs = set()
                for timestep in randomly_select_scenario.keys():
                    object_refs.update(randomly_select_scenario[timestep].keys())
                object_refs = sorted(object_refs)
                params_all = ""
                current_chunk = ""
                chunks = []
                tc = TokenCount(model_name="gpt-3.5-turbo")
                for time_step in range(self.total_time_step + 1):
                    junction_info = None
                    lane_info = None

                    params = ""
                    for object_ref in object_refs:
                        if f'timestep_{time_step}' in randomly_select_scenario and object_ref in randomly_select_scenario[f'timestep_{time_step}']:
                            configuration = randomly_select_scenario[f'timestep_{time_step}'][object_ref]
                            if isinstance(configuration, dict) and configuration is not None:
                                if 'position' in configuration:
                                    for parameter_object in ["position", "rotation", "velocity", "angular_velocity"]:
                                        params += f"The '{parameter_object}' of {object_ref} is ({configuration[parameter_object]['x']}, {configuration[parameter_object]['y']}, {configuration[parameter_object]['z']}).\n"

                                    if 'type' in configuration:
                                        params += f"The {object_ref} is a {configuration['type']}"
                                    
                                    if 'Collided_With_Ego' in configuration:
                                        if configuration['Collided_With_Ego'] == True:
                                            params += " and it collided with the Ego.\n"
                                            collided = True
                                        else:
                                            params += ".\n"
                                elif "Left_Boundary" in configuration:
                                    lane_info = configuration
                                elif "Junction_Position" in configuration:
                                    junction_info = configuration
                                elif "rain" in configuration:
                                    weather_info = configuration
                            elif object_ref == "time":
                                time_info = configuration      
                            else:
                                params += f"The '{object_ref}' is not present at {round(time_step * 0.5, 1)} seconds.\n"

                    params += f"Left Boundary is {lane_info['Left_Boundary']}.\n"
                    params += f"Right Boundary is {lane_info['Right_Boundary']}.\n"
                    params += f"Left Lane is in the {lane_info['Left_Lane_Direction']}.\n"
                    params += f"Right Lane is in the {lane_info['Right_Lane_Direction']}.\n"

                    if junction_info['Junction_Position'] != "out of":
                        params += f"Ego is {junction_info['Junction_Position']} the junction.\n"
                        if junction_info["Has_Horizontal_Left_Entry"] == True:
                            params += "This junction has horizontal left entry.\n"
                        if junction_info["Has_Horizontal_Right_Entry"] == True:
                            params += "This junction has horizontal right entry.\n"
                        if junction_info["Has_Vertical_Inverse_Entry"] == True:
                            params += "This junction has vertical inverse entry.\n"
                        if junction_info["Has_Vertical_Forward_Entry"] == True:
                            params += "This junction has vertical forward entry.\n"
                                    
                        else:
                            params += "Ego is not in the junction.\n"
                    if time_info == 10:
                        params += "The current time is morning.\n"
                    elif time_info == 14:
                        params += "The current time is afternoon.\n"
                    else:
                        params += "The current time is evening.\n"
                    params += "The weather conditions are as follows:\n"

                    if weather_info["rain"] == 0:
                        params += "Rain: None (clear weather)\n"
                    elif weather_info["rain"] == 0.2:
                        params += "Rain: Light rain\n"
                    elif weather_info["rain"] == 0.5:
                        params += "Rain: Moderate rain\n"
                    elif weather_info["rain"] == 1:
                        params += "Rain: Heavy rain\n"
                    if weather_info["fog"] == 0:

                        params += "Fog: No fog\n"
                    elif weather_info["fog"] == 0.2:
                        params += "Fog: Light fog\n"
                    elif weather_info["fog"] == 0.5:
                        params += "Fog: Moderate fog\n"
                    elif weather_info["fog"] == 1:
                        params += "Fog: Dense fog\n"

                    if weather_info["wetness"] == 0:
                        params += "Wetness: Dry\n"
                    elif weather_info["wetness"] == 0.2:
                        params += "Wetness: Slightly wet\n"
                    elif weather_info["wetness"] == 0.5:
                        params += "Wetness: Moderately wet\n"
                    elif weather_info["wetness"] == 1:
                        params += "Wetness: Fully wet\n"
                  
                    current_chunk += f"At {round(time_step * 0.5, 1)} seconds:\n{params}\n"
                    tokens = tc.num_tokens_from_string(current_chunk)
          
                    if tokens >= MAX_CHUNK_SIZE:
                        print(f"--------------Tokens in the string: {tokens}------------")
                        chunks.append(current_chunk)
                        current_chunk = ""
     
                    params_all += f"At {round(time_step * 0.5, 1)} seconds:\n{params}\n"

                if current_chunk:
                    chunks.append(current_chunk)
                chunk_len =(len(chunks))
                print(f"-----Chunks size: {chunk_len}-------")
                # sys.exit()
                messages = []
                sys_message = {}
                sys_message["role"] = "system"
                sys_message["content"] = """You know a lot about autonomous driving and some realistic autonomous driving crash datasets.
You are a helpful crash scenario realism evaluation assistant.
You will evaluate the following autonomous driving crash scenario, and check whether the scenario is realistic and natural. A collision is normal and you will evaluate realism of this case.
You will provide the corresponding realism score. The scale is 1.0-10.0, where 1.0 is unrealistic and 10.0 is realistic.
You know the scenario is from the LGSVL simulator.
If the scenario is long, it will be split into multiple parts, and you will receive a continuation prompt for each part. Please evaluate each part based on the same factors as before. 
Once I have sent all parts of the scenario, I will inform you by saying 'ALL PARTS SENT'.
Each part is considered a sub-scenario and should be evaluated independently based on its realism and naturalness.
Notes for evaluating the scenarios:
1. When the scenario mentions the ego vehicle colliding after each step, it may imply that the ego vehicle and another vehicle are stuck after the collision.
2. If the position of an NPC (Non-Player Character) is (0, 0, 0), it means the NPC has gone out of the map and no longer affects the scenario, so you do not need to evaluate it further.
3. The scenario is designed to test how the ego vehicle reacts, so do not judge the ego's reaction.
4. If any value of an NPC, such as velocity or angular_velocity, is (0, 0, 0) or too low, it may mean that these values cannot be estimated at that time and should not be treated as their actual values."""
                messages.append(sys_message)
                initial_prompt = {}
                initial_prompt["role"] = "user"
                initial_prompt[
                    "content"] = f"""{road_description[road]} The scenario contains {chunk_len} parts. This is the first part of a scenario generated for testing the ego vehicle. 
                    The scenario starts at 0.0 seconds, with all objects starting from rest. {chunks[0]} The scenario lasts for {self.total_time_step} time steps, each lasting 0.5 seconds.

Your task is to evaluate the degree of realism of this scenario on a scale from 1 to 10 based on the following factors:
- Physical and Environmental Accuracy
- Behavioral Realism of Other Road Users
- Dynamic Scenario Elements
- Edge Case Representation
- Temporal Consistency and Detail
For each factor, provide the realism score (1-10) and a brief reason for the score. After evaluating each factor, calculate the **Average Realism Score** (average of all individual scores) and determine if the scenario is **Realistic** (true or false) based on the overall evaluation.

Please provide the output in the following JSON format:

```json
{{
  "Physical and Environmental Accuracy": {{
    "realism_score": <score>,
    "reason": "<reason>"
  }},
  "Behavioral Realism of Other Road Users": {{
    "realism_score": <score>,
    "reason": "<reason>"
  }},
  "Dynamic Scenario Elements": {{
    "realism_score": <score>,
    "reason": "<reason>"
  }},
  "Edge Case Representation": {{
    "realism_score": <score>,
    "reason": "<reason>"
  }},
  "Temporal Consistency and Detail": {{
    "realism_score": <score>,
    "reason": "<reason>"
  }},
  "Average Realism Score": <average score>,
  "Realistic": <true or false>
}}"""           
                if len(chunks) == 1:
                    initial_prompt["content"] += "\nALL PARTS SENT."
                messages.append(initial_prompt)

                outputs = ""
                output = ""
                create_times = []
                output_times = []
                total_times = []
                            
                for index in range(repeat_time):
                    output, create_time, output_time, total_time = self.openai_gpt_4(model, messages, 0)
                    outputs = outputs + "-------------------Part 1----------------------\n" + output + "\n"
                    create_times.append(create_time)
                    output_times.append(output_time)
                    total_times.append(total_time)
                    for idx, chunk in enumerate(chunks[1:], start=2):
                        messages = []
                        messages.append(sys_message)
                        continuation_prompt = {
                            "role": "user",
                            "content": f"""
                   The scenario contains {chunk_len} parts. This is part {idx} of a scenario generated for testing the ego vehicle. Keep in mind that this is a continuation, and the previous context is important for your evaluation.
                   The scenario starts at 0.0 seconds. The scenario lasts for {self.total_time_step} time steps, each lasting 0.5 seconds.
                   {chunk}

                   Your task is to evaluate the degree of realism of this scenario on a scale from 1 to 10 based on the following factors:
                    - Physical and Environmental Accuracy
                    - Behavioral Realism of Other Road Users
                    - Dynamic Scenario Elements
                    - Edge Case Representation
                    - Temporal Consistency and Detail
                    For each factor, provide the realism score (1-10) and a brief reason for the score. After evaluating each factor, calculate the **Average Realism Score** (average of all individual scores) and determine if the scenario is **Realistic** (true or false) based on the overall evaluation.

                    Please provide the output in the following JSON format:

                    ```json
                    {{
                    "Physical and Environmental Accuracy": {{
                        "realism_score": <score>,
                        "reason": "<reason>"
                    }},
                    "Behavioral Realism of Other Road Users": {{
                        "realism_score": <score>,
                        "reason": "<reason>"
                    }},
                    "Dynamic Scenario Elements": {{
                        "realism_score": <score>,
                        "reason": "<reason>"
                    }},
                    "Edge Case Representation": {{
                        "realism_score": <score>,
                        "reason": "<reason>"
                    }},
                    "Temporal Consistency and Detail": {{
                        "realism_score": <score>,
                        "reason": "<reason>"
                    }},
                    "Average Realism Score": <average score>,
                    "Realistic": <true or false>
                    }}
                    
                    """
                        }
                        if idx == len(chunks):
                            continuation_prompt["content"] += "\nALL PARTS SENT."
                        messages.append(continuation_prompt)
                        output = ""
                        output, create_time, output_time, total_time = self.openai_gpt_4(model, messages, 0)
                        outputs = outputs + f"-------------------Part {idx}----------------------\n" + output + "\n"
                        create_times.append(create_time)
                        output_times.append(output_time)
                        total_times.append(total_time)
                    output = outputs
                    #save 
                    evaluate_R_MR_extra_4full_experiments_results_path = "./outputs_results/" + road + "-scenarios/" + scenario_name + "/"
                    if not os.path.exists(evaluate_R_MR_extra_4full_experiments_results_path):
                        os.makedirs(evaluate_R_MR_extra_4full_experiments_results_path)
                    evaluate_R_MR_extra_4full_experiments_results_file_name = evaluate_R_MR_extra_4full_experiments_results_path + scenario_name + \
                                                                                model.split("/")[
                                                                                    -1] + "_" + str(
                        index) + ".txt"
                    with open(evaluate_R_MR_extra_4full_experiments_results_file_name, 'w') as file:
                        file.write(f"model: {model}\n\n")
                        file.write(output + "\n\n\n")
                        file.write("----------------------Scenario---------------------\n" + params_all + "\n\n\n")
                        file.write(
                            f"create_time: {create_time}s output_time: {output_time}s total_time: {total_time}s\n")


                    print(
                        road + " " + scenario_name + "===================================================")
                    print(f"model: {model}")
                    print(f"index: {index}\n")

                 
if __name__ == '__main__':
    llmapi = LLMAPI()

    model = "gpt-4o-mini"
    repeat_time = 1
    llmapi.evaluate_R_MR_extra_4full_experiments(model, repeat_time, 1)