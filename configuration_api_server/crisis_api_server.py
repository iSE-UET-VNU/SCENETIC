import socket
import time
import os
import json
import numpy as np
import math
import threading
import random
import queue
import pickle
import torch
import requests

import lgsvl
from lgsvl.agent import NpcVehicle
from lgsvl.dreamview import CoordType

from flask import Flask, request

from utils import pedestrian, npc_vehicle, calculate_measures, calculate_distance
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

from datetime import timedelta

from server_constants import *

script_dir = os.path.dirname(os.path.abspath(__file__))

observation_time = 6 

# app config
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

sim = lgsvl.Simulator(os.environ.get(
    "SIMUSaveStateLATOR_HOST", SIMULATOR_HOST), SIMULATOR_PORT)

DREAMVIEW = None

probability = 0
ETTC = 100
DTO = 100000
JERK = 100

collision_object = None
collision_tag = False
collision_uid = "No collision"

EGO = None
NPC_QUEUE = queue.Queue(maxsize=10)

is_collision_ahead = False
prev_acc = 0
time_offset = TIME_OFFSET
pedes_prev_pos = {}

scenario = {}
time_step_counter = 0
scenario_counter = 0
current_road = 0

SAVING_SCENARIO = False

current_lane = {
    'left_boundary': {
        'a': 0,
        'b': 0,
        'c': 0
    },
    'right_boundary': {
        'a': 0,
        'b': 0,
        'c': 0
    },
    'left_boundary_type': 0,
    'right_boundary_type': 0,
    'left_lane_direction': 0,
    'right_lane_direction': 0,
    'id': "lane_54"
}

cars = ['Jeep', 'BoxTruck', 'Sedan', 'SchoolBus', 'SUV', 'Hatchback']
colors = ['pink', 'yellow', 'red', 'white', 'black', 'skyblue']

api_prefix = '/crisis/'

# setup connect to apollo
msg_socket = None

# traffic condiition map
map = MAP # map: tartu, sanfrancisco, borregasave
lanes_map = None
junctions_map = None
lanes_junctions_map = None

def connect_to_apollo_listener():
    
    global msg_socket
    
    print("Setup connection to apollo extractor")
    
    msg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (APOLLO_HOST, EXTRACTOR_PORT)

    msg_socket.connect(server_address)

def load_map_traffic_condition():
    
    global lanes_map
    global junctions_map
    global lanes_junctions_map
    
    lanes_map_file = f"{script_dir}/map/{map}_lanes.pkl"

    with open(lanes_map_file, "rb") as file:
        lanes_map = pickle.load(file)

    file.close()

    junctions_map_file = f"{script_dir}/map/{map}_junctions.pkl"

    with open(junctions_map_file, "rb") as file2:
        junctions_map = pickle.load(file2)

    file2.close()

    lanes_junctions_map_file = f"{script_dir}/map/{map}_lanes_junctions.pkl"

    with open(lanes_junctions_map_file, "rb") as file3:
        lanes_junctions_map = pickle.load(file3)

    file3.close()

# function raised when a collision happened
def on_collision(agent1, agent2, contact):
    name1 = agent1.__dict__.get('name')
    name2 = agent2.__dict__.get('name') if agent2 is not None else "OBSTACLE"
    uid = agent2.__dict__.get('uid') if agent2 is not None else "OBSTACLE"
    print("{} collided with {} at {}".format(name1, name2, contact))

    global collision_tag
    global collision_object
    global collision_uid
    global is_collision_ahead
    
    collision_tag = True
    collision_uid = uid
    collision_object = name2
    
    try:
        # judge pedestrian collided ahead of ego (calculate line equation and compare pedestrian and ego's position)
        if name2 != 'OBSTACLE' and not isinstance(agent2, NpcVehicle):
            ego_pos = agent1.state.position
            ego_rot = agent1.state.rotation
            
            yaw_rad = np.deg2rad(ego_rot.y)
            
            ego_dir = ego_rot
    
            ego_dir.x = np.sin(yaw_rad)
            ego_dir.z = np.cos(yaw_rad)
            
            length = 4.7
            
            vel_ = math.sqrt(ego_dir.x ** 2 + ego_dir.z ** 2)
            ahd_pt = ego_pos + (length / 2) / vel_ * ego_dir 
            
            a = ego_dir.x 
            b = ego_dir.z
            c = - a * ahd_pt.x - b * ahd_pt.z
            
            pedes_pos = agent2.state.position
            
            val_ego = a * ego_pos.x + b * ego_pos.z +c
            val_pedes = a * pedes_pos.x + b * pedes_pos.z + c
            
            if val_ego * val_pedes <= 0:
                is_collision_ahead = True            
            
    except KeyError:
        print('Error in identifying collision position')

# calculate position's value with lane boundary line equation
def get_boundary_value(boundary, position):
    return boundary['a'] * position.x + boundary['b'] * position.z + boundary['c']

# check whether generated vehicle conflict to others
def get_no_conflict_position(position, car):
    if car == 'BoxTruck' or car == 'SchoolBus':
        sd = 10
    else:
        sd = 8
        
    generate = True
    if CONFLICT_CONSTRAIN:
        
        agents = sim.get_agents()
        for agent in agents:

            # check whether ego and generated vehicle are in the same lane or not
            left_val_1 = get_boundary_value(current_lane['left_boundary'], position)
            right_val_1 = get_boundary_value(current_lane['right_boundary'], position)

            left_val_2 = get_boundary_value(current_lane['left_boundary'], agent.transform.position)
            right_val_2 = get_boundary_value(current_lane['right_boundary'], agent.transform.position)

            is_considered = ((left_val_1 * left_val_2) > 0 and (right_val_1 * right_val_2) > 0)

            if is_considered:
                distance = calculate_distance([position.x, position.y, position.z], [agent.transform.position.x, agent.transform.position.y, agent.transform.position.z])
                if distance < sd: # ego and vehicle are in the same lane and conflict each other 
                    generate = False
                    break

    return generate

# set vehicles's color
def set_color(color):
    color_v = lgsvl.Vector(0, 0, 0)
    if color == 'black':
        color_v = lgsvl.Vector(0, 0, 0)
    elif color == 'white':
        color_v = lgsvl.Vector(1, 1, 1)
    elif color == 'yellow':
        color_v = lgsvl.Vector(1, 1, 0)
    elif color == 'pink':
        color_v = lgsvl.Vector(1, 0, 1)
    elif color == 'skyblue':
        color_v = lgsvl.Vector(0, 1, 1)
    elif color == 'red':
        color_v = lgsvl.Vector(1, 0, 0)
    elif color == 'green':
        color_v = lgsvl.Vector(0, 1, 0)
    elif color == 'blue':
        color_v = lgsvl.Vector(0, 0, 1)
    return color_v

# control number of agents
def control_agents_density(agent):
    if CONTROL_DENSITY:
        if NPC_QUEUE.full():
            sim.remove_agent(NPC_QUEUE.get())
            NPC_QUEUE.put(agent)
        else:
            NPC_QUEUE.put(agent)

def save_scenario(agent_uid, ego_state, npc_state, is_npc_vehicle):
    global scenario
    global time_step_counter
    global collision_uid
    global current_lane
    
    boundary_type = ['Unknown', 'Dotted Yellow', 'Dotted White', 'Solid Yellow', 'Solid White', 'Double Yellow', 'Curb']
    direction_type = ['None', 'Same Direction to Ego', 'Opposite Direction to Ego']
    weather_severity_type = {
        NONE_THRESHOLD: 'Nice', 
        LIGHT_THRESHOLD:'Light', 
        MODERATE_THRESHOLD: 'Moderate', 
        HEAVY_THRESHOLD: 'Heavy'
    }
    
    # Traffic Signal Info
    signal = interpreter_signal(sim.get_controllable(ego_state.position, "signal").current_state)
    
    if signal == -1:
        scenario["timestep_" + str(time_step_counter)]['Ahead_Traffic_Signal'] = "Red"
    elif signal == 0:
        scenario["timestep_" + str(time_step_counter)]['Ahead_Traffic_Signal'] = "Yellow"
    elif signal == 1:
        scenario["timestep_" + str(time_step_counter)]['Ahead_Traffic_Signal'] = "Green"
    else:
        scenario["timestep_" + str(time_step_counter)]['Ahead_Traffic_Signal'] = "None"
        
    # Weather Info
    weather_state = sim.weather
    rain_info = round(weather_state.rain, 1)
    fog_info = round(weather_state.fog, 1)
    wetness_info = round(weather_state.wetness, 1)
    scenario["timestep_" + str(time_step_counter)]['weather'] = {
        'rain_level': weather_severity_type[str(rain_info)],
        'fog_level': weather_severity_type[str(fog_info)],
        'wetness_level': weather_severity_type[str(wetness_info)]
    }    
    
    # Time of day Info
    time_of_day = (sim.time_of_day - time_offset + 24) % 24
    scenario["timestep_" + str(time_step_counter)]['time_of_day'] = {
        "time": time_of_day
    }
    
    # Ego States
    scenario["timestep_" + str(time_step_counter)]['Ego'] = {}

    scenario["timestep_" + str(time_step_counter)]['Ego'] = {
        'position': {
            'x': round(ego_state.position.x, 3),
            'y': round(ego_state.position.y, 3),
            'z': round(ego_state.position.z, 3)
        },
        'rotation': {
            'x': round(ego_state.rotation.x, 3),
            'y': round(ego_state.rotation.y, 3),
            'z': round(ego_state.rotation.z, 3),
        },
        'velocity': {
            'x': round(ego_state.velocity.x, 3),
            'y': round(ego_state.velocity.y, 3),
            'z': round(ego_state.velocity.z, 3)
        },
        'angular_velocity': {
            'x': round(ego_state.angular_velocity.x, 3),
            'y': round(ego_state.angular_velocity.y, 3), 
            'z': round(ego_state.angular_velocity.z, 3)
        }
    }

    # Dynamic Obstacles States
    for i in range(0, len(npc_state)):
        npc_id = 'NPC' + agent_uid[i + 1]
        npc_id = npc_id[0:int(len(npc_id) / 2)]
        scenario["timestep_" + str(time_step_counter)][npc_id] = {
            'position': {
                'x': round(npc_state[i].position.x, 3),
                'y': round(npc_state[i].position.y, 3), 
                'z': round(npc_state[i].position.z, 3)
            },
            'rotation': {
                'x': round(npc_state[i].rotation.x, 3),
                'y': round(npc_state[i].rotation.y, 3),
                'z': round(npc_state[i].rotation.z, 3)
            },
            'velocity': {
                'x': round(npc_state[i].velocity.x, 3),
                'y': round(npc_state[i].velocity.y, 3),
                'z': round(npc_state[i].velocity.z, 3)
            },
            'angular_velocity': {
                'x': round(npc_state[i].angular_velocity.x, 3),
                'y': round(npc_state[i].angular_velocity.y, 3),
                'z': round(npc_state[i].angular_velocity.z, 3)
            }
        }
        
        if is_npc_vehicle[i]:
            scenario["timestep_" + str(time_step_counter)][npc_id]['type'] = "Vehicle"
        else:
            scenario["timestep_" + str(time_step_counter)][npc_id]['type'] = 'Pedestrian'
            
        if agent_uid[i + 1] == collision_uid:
            scenario["timestep_" + str(time_step_counter)][npc_id]["Collided_With_Ego"] = True
        else:
            scenario["timestep_" + str(time_step_counter)][npc_id]["Collided_With_Ego"] = False
        
        
    # Lane Info        
    scenario["timestep_" + str(time_step_counter)]['Lane'] = {
        'Left_Boundary': boundary_type[current_lane['left_boundary_type']],
        'Right_Boundary': boundary_type[current_lane['right_boundary_type']],
        'Left_Lane_Direction': direction_type[current_lane['left_lane_direction']],
        'Right_Lane_Direction': direction_type[current_lane['right_lane_direction']]
    }
    
    # Junction
    transform = lgsvl.Transform(
        lgsvl.Vector(ego_state.position.x, ego_state.position.y,
                     ego_state.position.z), lgsvl.Vector(ego_state.rotation.x, ego_state.rotation.y, ego_state.rotation.z)
    )
    gps = sim.map_to_gps(transform)
    dest_x = gps.easting
    dest_y = gps.northing

    transform_next = lgsvl.Transform(
        lgsvl.Vector(ego_state.position.x + ego_state.velocity.x, ego_state.position.y + ego_state.velocity.y,
                     ego_state.position.z + ego_state.velocity.z), lgsvl.Vector(ego_state.rotation.x, ego_state.rotation.y, ego_state.rotation.z)
    )
    gps_next = sim.map_to_gps(transform_next)
    dest_x_next = gps_next.easting
    dest_y_next = gps_next.northing

    ego_direction = {
        'x': dest_x_next - dest_x,
        'y': dest_y_next - dest_y
    }

    junction_id, junction_position = is_in_junction([dest_x, dest_y])
    
    print("Junction id: ", junction_id, "Position: ", junction_position) 
    
    has_vertical_entry_inverse = False
    has_vertical_entry_forward = False
    has_horizontal_entry_left = False
    has_horizontal_entry_right = False
    
    if junction_id is not None:
        for lane_id in lanes_junctions_map[junction_id]:
            lane_vector = {
                'x': lanes_map[lane_id]['central_curve'][1]['x'] - lanes_map[lane_id]['central_curve'][0]['x'],
                'y': lanes_map[lane_id]['central_curve'][1]['y'] - lanes_map[lane_id]['central_curve'][0]['y']
            } 

            dis_vector = {
                'x': lanes_map[lane_id]['central_curve'][0]['x'] - dest_x,
                'y': lanes_map[lane_id]['central_curve'][0]['y'] - dest_y
            }    
            
            _, direct = entry_detail(ego_direction, lane_vector, dis_vector)
            
            if direct == 'left':
                has_horizontal_entry_left = True
            elif direct == 'right':
                has_horizontal_entry_right = True
            elif direct == 'inverse':
                has_vertical_entry_inverse = True
            else:
                has_vertical_entry_forward = True
                
    scenario["timestep_" + str(time_step_counter)]['Junction'] = {
        'Has_Horizontal_Left_Entry': has_horizontal_entry_left,
        'Has_Horizontal_Right_Entry': has_horizontal_entry_right,
        'Has_Vertical_Inverse_Entry': has_vertical_entry_inverse,
        'Has_Vertical_Forward_Entry': has_vertical_entry_forward
    }

    if junction_position == 0:
        scenario["timestep_" + str(time_step_counter)]['Junction']["Junction_Position"] = "out of"
    elif junction_position == 1:
        scenario["timestep_" + str(time_step_counter)]['Junction']["Junction_Position"] = "near"
    else:
        scenario["timestep_" + str(time_step_counter)]['Junction']["Junction_Position"] = "in"


# calculate measures thread, use in multi-thread
def calculate_measures_thread(npc_state, ego_state, is_npc_vehicle, ttc_list, distance_list, probability_list, collision_tag_=False):

    TTC, distance, probability2 = calculate_measures(npc_state, ego_state, is_npc_vehicle)
    
    ttc_list.append(round(TTC, 6))

    distance_list.append(round(distance, 6))
    
    if collision_tag_:
        probability2 = 1
    probability_list.append(round(probability2, 6))   
        
# calculate metrics
def calculate_metrics(agents, ego, uid = None):
    
    global probability
    global ETTC
    global DTO
    global JERK
    
    global collision_tag
    global collision_object
    global collision_uid
    
    global is_collision_ahead
    global prev_acc
    global pedes_prev_pos
    
    global scenario
    global time_step_counter

    collision_object = None
    collision_uid = "No collision"

    is_collision_ahead = False
    
    spd_bf_col = 0
    is_first_collision = True
    pedes_mov_fw_to = False

    print("Calculating metrics ....")

    probability_list = []
    ettc_list = []
    distance_list = []
    jerk_list = []
    
    i = 0
    time_step = 0.5
    sliding_step = 0.25
    
    sudden_appearance = False
    overlapping = False
    position_list = {}

    while i < observation_time / time_step:

        if SAVING_SCENARIO:
            scenario['timestep_' + str(time_step_counter)] = {}
        
        # check apollo's modules status from dreamview
        check_modules_status()
        
        # run simulator
        for k in range(0, int(time_step / sliding_step)):
            
            spd_bf_col = ego.state.speed
            
            sim.run(time_limit=sliding_step)  # , time_scale=2
            
            collision_info_ = str(collision_object)
            collision_type_ = 'None'
            
            if collision_info_ == 'OBSTACLE':
                collision_type_ = "obstacle"
            if collision_info_ in npc_vehicle:
                collision_type_ = "npc_vehicle"
            if collision_info_ in pedestrian:
                collision_type_ = "pedestrian"
            
            if collision_tag and is_first_collision:
                print("Ego speed before collision: ", spd_bf_col)
                is_first_collision = False
                
                # if ego's speed < 1.0 and it collided with pedestrain, it means pedestrian deliberately moved forward to ego
                if spd_bf_col < 1.0 and collision_type_ == 'pedestrian':
                    pedes_mov_fw_to = True
            
            if uid and collision_tag:
                print("Collision at: ", (i * 2 + k) * 0.25, "-", (i * 2 + k + 1) * 0.25)
                
                # if a vehicle generated and colliding to ego after less than 0.75 seconds, it means that this collision is beyond ego's processing capacity
                if uid == collision_uid and (i * 2 + k + 1) * 0.25 <= 0.75:
                    sudden_appearance = True
                        
        if SAVING_SCENARIO:
            local_info, control_info = get_apollo_msg_for_saving()
            extract_lane_info(local_info, control_info)
        
        # calculate jerk
        # ego_acc = get_ego_acceleration()
        ego_acc = 5
        jerk_list.append(abs(ego_acc - prev_acc) / 0.5)
        prev_acc = ego_acc
        
        npc_state = []
        is_npc_vehicle = []
        agent_uid = []
        pos = {}
        
        agent_uid.append(ego.uid)
                
        ego_state = ego.state
        
        for j in range(1, len(agents)):
            state_ = agents[j].state
            is_npc = (isinstance(agents[j], NpcVehicle))
            is_npc_vehicle.append(is_npc)
            agent_uid.append(agents[j].uid)
            
            if not is_npc:
                pedes_uid = agents[j].uid
                if pedes_uid in pedes_prev_pos:
                    state_.velocity.x = (state_.transform.position.x - pedes_prev_pos[pedes_uid][0]) / 0.5
                    state_.velocity.y = (state_.transform.position.y - pedes_prev_pos[pedes_uid][1]) / 0.5
                    state_.velocity.z = (state_.transform.position.z - pedes_prev_pos[pedes_uid][2]) / 0.5
                
                pedes_prev_pos[pedes_uid] = np.array([state_.transform.position.x, state_.transform.position.y, state_.transform.position.z])
                
            pos[agents[j].uid] = {
                'x': state_.position.x,
                'y': state_.position.y,
                'z': state_.position.z,
                'dis_to_ego': math.sqrt((state_.position.x - ego_state.position.x) ** 2 + (state_.position.y - ego_state.position.y) ** 2 + (state_.position.z - ego_state.position.z) ** 2)
            }
            
            # if altitude gap is more than 0.4, vehicle lies on top of ego
            if not overlapping and (state_.position.x != 0 or state_.position.y != 0 or state_.position.z != 0):
                if abs(ego_state.position.y - state_.position.y) > 0.4 and agents[j].uid == collision_uid:
                    overlapping = True
                
            npc_state.append(state_)
        
        pos[ego.uid] = {
            'x': ego_state.position.x,
            'y': ego_state.position.y,
            'z': ego_state.position.z,
            'dis_to_ego': 0
        }

        # save to scenario.json
        if SAVING_SCENARIO:
            save_scenario(agent_uid, ego_state, npc_state, is_npc_vehicle)
            time_step_counter += 1
            
        position_list[str(i)] = pos

        thread = threading.Thread(
            target=calculate_measures_thread,
            args=(npc_state, ego_state, is_npc_vehicle, ettc_list, 
                  distance_list, probability_list, collision_tag,)
        )

        thread.start()

        if collision_tag:
            collision_tag = False

        i += 1
        
        if i == int(observation_time / time_step):
            time.sleep(0.5)

    collision_uid_, is_collision_ahead_ = get_collision_info()
    
    ETTC = round(min(ettc_list), 6)
    DTO = round(min(distance_list), 6)
    JERK = round(max(jerk_list), 6)
    probability = round(max(probability_list), 6)
    
    return {'ETTC': ettc_list, 'distance': distance_list, 'JERK': jerk_list, 'collision_uid': collision_uid_, 'probability': probability_list, "sudden_appearance": sudden_appearance, 
            "overlapping": overlapping, 'position_list': position_list, 'generated_uid': uid, 'is_collision_ahead': is_collision_ahead_, 'pedes_mov_fw_to': pedes_mov_fw_to} 


@app.route(f'{api_prefix}/ego/collision_info', methods=['GET'])
def get_collision_info():
    
    global collision_uid
    global is_collision_ahead
    
    is_collision_ahead_ = is_collision_ahead
    
    return collision_uid, is_collision_ahead_

@app.route(f'{api_prefix}/set-observation-time', methods=['POST'])
def set_observation_time():
    
    global observation_time
    
    observation_time = int(request.args.get('observation_time'))
    print(observation_time)
    return 'get time'

@app.route(f'{api_prefix}/load-scene', methods=['POST'])
def load_scene():
    
    global EGO
    global prev_lane_id
    global DREAMVIEW
    
    global prev_acc
    global time_offset
    global pedes_prev_pos
    
    global scenario
    global time_step_counter
    global scenario_counter
    global current_road
    global SAVING_SCENARIO
    global NPC_QUEUE

    while not NPC_QUEUE.empty():
        NPC_QUEUE.get_nowait()

    # saving scenario
    saving = request.args.get('saving')
    
    if saving == "1":
        SAVING_SCENARIO = True
    else:
        SAVING_SCENARIO = False

    if 'timestep_0' in scenario and SAVING_SCENARIO:

        print(scenario)

        prefix = script_dir + "/" + "road" + str(current_road) + "-scenarios/"

        os.makedirs(prefix, exist_ok=True)
        
        scenario_path = prefix + "scenario" + str(scenario_counter) + ".json"
        with open(scenario_path, "w") as file:
            json.dump(scenario, file, indent=4)
        
        scenario_counter += 1

    scenario = {}
    time_step_counter = 0
    
    prev_acc = 0
    prev_lane_id = ""
    pedes_prev_pos = {}

    # load scene
    print(f'Observation time: {observation_time}')
    scene = str(request.args.get('scene'))
    road_num = str(request.args.get('road_num'))
    current_road = int(road_num)
    
    if sim.current_scene == scene:
        sim.reset()
    else:
        sim.load(scene)

    # set ego vehicle
    EGO = None
    state = lgsvl.AgentState()
    
    endpoint_json = open(f'{script_dir}/map_endpoint/ego_endpoint.json', 'r')
    endpoint_list = endpoint_json.read()
    ego_endpoint = json.loads(s=endpoint_list)
    
    start_point = ego_endpoint[scene]['road' + road_num]['start']
    
    state.transform.position.x = start_point['position']['x']
    state.transform.position.y = start_point['position']['y']
    state.transform.position.z = start_point['position']['z']
    state.transform.rotation.x = start_point['rotation']['x']
    state.transform.rotation.y = start_point['rotation']['y']

    forward = lgsvl.utils.transform_to_forward(state.transform)
    state.velocity = 3 * forward

    EGO = sim.add_agent(EGO_UID, lgsvl.AgentType.EGO, state)
    EGO.connect_bridge(os.environ.get("BRIDGE_HOST", APOLLO_HOST), BRIDGE_PORT)
    DREAMVIEW = lgsvl.dreamview.Connection(sim, EGO, APOLLO_HOST, str(DREAMVIEW_PORT))

    sim.get_agents()[0].on_collision(on_collision)
    sim.set_time_of_day((10 + time_offset) % 24, fixed=True)

    # set destination
    end_point = ego_endpoint[scene]['road' + road_num]['end']
    requests.post(f"http://{API_SERVER_HOST}:{API_SERVER_PORT}/crisis/set-destination?des_x={end_point['position']['x']}&des_y={end_point['position']['y']}&des_z={end_point['position']['z']}")

    sim.run(2)

    return 'load success'

@app.route(f'{api_prefix}/run', methods=['POST'])
def run():
    sim.run(8)
    return 'sim run'

@app.route(f'{api_prefix}/set-destination', methods=['POST'])
def set_destination():
    
    global EGO
    global DREAMVIEW
    x = float(request.args.get('des_x'))
    y = float(request.args.get('des_y'))
    z = float(request.args.get('des_z'))
    DREAMVIEW.set_destination(x, z, y, coord_type=CoordType.Unity)
    return 'set destination.'

@app.route(f'{api_prefix}/control/weather/nice', methods=['POST'])
def nice():
    sim.weather = lgsvl.WeatherState(rain=0, fog=0, wetness=0)
    
    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)


@app.route(f'{api_prefix}/control/weather/rain', methods=['POST'])
def rain():
    
    rain_level = request.args.get('rain_level')
    r_level = 0
    w_level = 0
    if rain_level == 'light':
        r_level = 0.2
        w_level = 0.2

    elif rain_level == 'moderate':
        r_level = 0.5
        w_level = 0.5
        
    elif rain_level == 'heavy':
        r_level = 1
        w_level = 1
        
    sim.weather = lgsvl.WeatherState(rain=r_level, fog=0, wetness=w_level)

    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)


@app.route(f'{api_prefix}/control/weather/fog', methods=['POST'])
def fog():
   
    fog_level = request.args.get('fog_level')
    f_level = 0
    
    if fog_level == 'light':
        f_level = 0.2
        
    elif fog_level == 'moderate':
        f_level = 0.5
        
    elif fog_level == 'heavy':
        f_level = 1
        
    sim.weather = lgsvl.WeatherState(rain=0, fog=f_level, wetness=0)
    
    agents = sim.get_agents()
    ego = agents[0]
    
    return calculate_metrics(agents, ego)


@app.route(f'{api_prefix}/control/weather/wetness', methods=['POST'])
def wetness():
    
    wetness_level = request.args.get('wetness_level')
    w_level = 0
    if wetness_level == 'light':
        w_level = 0.2
        
    elif wetness_level == 'moderate':
        w_level = 0.5
        
    elif wetness_level == 'heavy':
        w_level = 1
        
    sim.weather = lgsvl.WeatherState(rain=0, fog=0, wetness=w_level)
    
    agents = sim.get_agents()
    ego = agents[0]

    return calculate_metrics(agents, ego)

@app.route(f'{api_prefix}/control/time-of-day', methods=['POST'])
def time_of_day():

    global time_offset
    
    time = request.args.get('time_of_day')
    day_time = (10 + time_offset) % 24  # initial time: 10
    
    if time == 'morning':
        day_time = (10 + time_offset) % 24
        
    elif time == 'noon':
        day_time = (14 + time_offset) % 24 
        
    elif time == 'evening':
        day_time = (20 + time_offset) % 24
        
    sim.set_time_of_day(day_time, fixed=True)

    agents = sim.get_agents()
    ego = agents[0]
    
    return calculate_metrics(agents, ego)

@app.route(f'{api_prefix}/agents/npc-vehicle/cross-road', methods=['POST'])
def add_npc_cross_road():

    global cars
    global colors
    global NPC_QUEUE
    
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    
    change_lane = int(request.args.get('maintainlane'))
    lane = request.args.get('lane')
    distance = str(request.args.get('position'))
    color_v = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    right = lgsvl.utils.transform_to_right(ego_transform)
    forward = lgsvl.utils.transform_to_forward(ego_transform)

    point = None
    if lane == 'right': 
        if distance == 'near':
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position + 4 * right + 17 * forward
            else:
                point = ego_transform.position + 4 * right + 12 * forward
        else:
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position + 4 * right + 30 * forward
            else:
                point = ego_transform.position + 4 * right + 25 * forward
    else:
        if distance == 'near':
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position - 4 * right + 17 * forward
            else:
                point = ego_transform.position - 4 * right + 12 * forward
        else:
            if which_car == 'BoxTruck' or which_car == 'SchoolBus':
                point = ego_transform.position - 4 * right + 30 * forward
            else:
                point = ego_transform.position - 4 * right + 25 * forward

    state = lgsvl.AgentState()
    state.transform = sim.map_point_on_lane(point)

    npc = None

    generate = get_no_conflict_position(state.position, which_car)
    if not generate:    
        if distance == 'near':
            point -= forward * 10
            state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(state.position, which_car)
            
        else:
            point += forward * 20
            state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(state.position, which_car)
           
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, state, color_v)
        npc.follow_closest_lane(True, 20)
        if change_lane:
            if lane == 'left':
                npc.change_lane(False)
            elif lane == 'right':
                npc.change_lane(True)

        control_agents_density(npc)
        
    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(f'{api_prefix}/agents/pedestrian/cross-road', methods=['POST'])
def add_pedestrian_cross_road():
    
    global pedes_prev_pos
    global NPC_QUEUE
    
    direction = request.args.get('direction')
    
    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    right = lgsvl.utils.transform_to_right(ego_transform)

    npc_state = lgsvl.AgentState()

    if direction == 'left':
        offset = - 5.0 * right
    else:
        offset = 5.0 * right

    wp = [lgsvl.WalkWaypoint(sim.map_point_on_lane(ego_transform.position + offset + 40 * forward).position, 1),
          lgsvl.WalkWaypoint(sim.map_point_on_lane(ego_transform.position - offset + 40 * forward).position, 1)]

    npc_state.transform.position = sim.map_point_on_lane(ego_transform.position + offset + 40.0 * forward).position

    generate = get_no_conflict_position(
        npc_state.transform.position, 'pedestrian')
    
    p = None
    
    if generate:
        name = pedestrian[random.randint(0, 8)]
        p = sim.add_agent(name, lgsvl.AgentType.PEDESTRIAN, npc_state)
        
        pedes_uid = p.uid
        pedes_prev_pos[pedes_uid] = np.array([npc_state.transform.position.x, npc_state.transform.position.y, npc_state.transform.position.z])
        
        p.follow(wp, loop=False)
        
        control_agents_density(p)
        
    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if p:
        uid = p.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(f'{api_prefix}/agents/npc-vehicle/drive-ahead', methods=['POST'])
def add_npc_drive_ahead():

    global cars
    global colors
    global NPC_QUEUE
    
    which_lane = request.args.get('lane')
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    color_v = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    forward_ = forward
    right = lgsvl.utils.transform_to_right(ego_transform)

    if distance == 'near':
        offset = 10
        if which_car == 'BoxTruck' or which_car == 'SchoolBus':
            forward = 15 * forward
            right = 4 * right
            speed = 10
        else:
            forward = 12 * forward
            right = 4 * right
            speed = 10
    else:
        offset = 50
        forward = 50 * forward
        right = 4 * right
        speed = 10
        

    if which_lane == "left":
        point = ego_transform.position - right + forward
    elif which_lane == "right":
        point = ego_transform.position + right + forward
    elif which_lane == "current":
        point = ego_transform.position + forward
    else:
        point = lgsvl.Vector(ego_transform.position.x + offset * math.cos(0), ego_transform.position.y,
                             ego_transform.position.z + offset * math.sin(0))

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = False
    generate = get_no_conflict_position(npc_state.position, which_car)

    if not generate:
        if distance == 'near':
            point += forward_ * 10
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            
        else:
            point -= forward_ * 20
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
        
    npc = None

    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, color_v)
        npc.follow_closest_lane(True, speed)
        if change_lane:
            if which_lane == 'left':
                npc.change_lane(False)
            elif which_lane == 'right':
                npc.change_lane(True)
            else:
                npc.change_lane(random.choice([True, False]))

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(f'{api_prefix}/agents/npc-vehicle/overtake', methods=['POST'])
def add_npc_overtake():
    
    global cars
    global colors
    global NPC_QUEUE
    
    which_lane = request.args.get('lane')
    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    color_v = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    forward_ = forward
    right = lgsvl.utils.transform_to_right(ego_transform)

    if which_car == 'BoxTruck' or which_car == 'SchoolBus':
        offset = 10
        forward = 10 * forward
        right = 4 * right
        speed = 20
    else:
        offset = 10
        forward = 5 * forward
        right = 4 * right
        speed = 20

    if which_lane == "left":
        point = ego_transform.position - right - forward
    elif which_lane == "right":
        point = ego_transform.position + right - forward
    elif which_lane == "current":
        point = ego_transform.position - forward
    else:
        point = lgsvl.Vector(ego_transform.position.x + offset * math.cos(0), ego_transform.position.y,
                             ego_transform.position.z + offset * math.sin(0))

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(npc_state.position, which_car)

    if not generate:    
        point -= forward_ * 10
        npc_state.transform = sim.map_point_on_lane(point)
        generate = get_no_conflict_position(npc_state.position, which_car)
        
    npc = None

    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, color_v)
        npc.follow_closest_lane(True, speed)
        if change_lane:
            if which_lane == 'left':
                npc.change_lane(False)
            elif which_lane == 'right':
                npc.change_lane(True)
            else:
                npc.change_lane(random.choice([True, False]))

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)


@app.route(f'{api_prefix}/agents/npc-vehicle/drive-opposite', methods=['POST'])
def add_npc_drive_opposite():

    global NPC_QUEUE
    global cars
    global colors

    which_car = cars[random.randint(0, 5)]
    color = colors[random.randint(0, 5)]
    change_lane = int(request.args.get('maintainlane'))
    distance = str(request.args.get('position'))
    color_v = set_color(color)

    ego_transform = sim.get_agents()[0].state.transform
    forward = lgsvl.utils.transform_to_forward(ego_transform)
    forward_ = forward
    right = lgsvl.utils.transform_to_right(ego_transform)

    if distance == 'near':
        offset = 20
    else:
        offset = 50
   
    forward = offset * forward
    right = 8 * right
    speed = 20

    point = ego_transform.position - right + forward

    npc_state = lgsvl.AgentState()
    npc_state.transform = sim.map_point_on_lane(point)

    generate = get_no_conflict_position(npc_state.position, which_car)
    if not generate:    
        if distance == 'near':
            point += forward_ * 10
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            
        else:
            point -= forward_ * 20
            npc_state.transform = sim.map_point_on_lane(point)
            generate = get_no_conflict_position(npc_state.position, which_car)
            
    npc = None
            
    if generate:
        npc = sim.add_agent(which_car, lgsvl.AgentType.NPC, npc_state, color_v)
        npc.follow_closest_lane(True, speed)
        npc.change_lane(change_lane == 1)

        control_agents_density(npc)

    agents = sim.get_agents()
    ego = agents[0]
    
    uid = None
    if npc:
        uid = npc.uid
    
    return calculate_metrics(agents, ego, uid)

def interpreter_signal(signal_state):
    code = 0
    if signal_state == 'red':
        code = -1
    elif signal_state == 'yellow':
        code = 0
    elif signal_state == 'green':
        code = 1
    return code

def get_ego_acceleration():
    global msg_socket

    msg_socket.send(json.dumps(["get_acc"]).encode("utf-8"))
    data = msg_socket.recv(1024)

    data = json.loads(data.decode("utf-8"))

    acceleration = data["acceleration"]

    return acceleration

def get_apollo_msg():
    global msg_socket

    msg_socket.send(json.dumps(["start_getting_data"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    local_info = data["local_info"]
    per_info = data["per_info"]

    return local_info, per_info, control_info

def get_apollo_msg_for_saving():
    global msg_socket

    msg_socket.send(json.dumps(["start_getting_data"]).encode("utf-8"))
    data = msg_socket.recv(2048)

    data = json.loads(data.decode("utf-8"))

    control_info = data["control_info"]
    local_info = data["local_info"]

    return local_info, control_info


def check_modules_status():
    global EGO
    global DREAMVIEW
    
    modules_status = DREAMVIEW.get_module_status()
    
    stop = False
    
    stop_time = 300
    
    if modules_status['Localization'] == False:
        print(20*'*', 'LOCALIZATION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Localization')
        print(20*'*', 'LOCALIZATION ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Prediction'] == False:
        print(20*'*', 'PREDICTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Prediction')
        print(20*'*', 'PREDICTION ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Transform'] == False:
        print(20*'*', 'TRANSFORM STOPPED', 20*'*')
        DREAMVIEW.enable_module('Transform')
        print(20*'*', 'TRANSFORM ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Control'] == False:
        print(20*'*', 'CONTROL STOPPED', 20*'*')
        DREAMVIEW.enable_module('Control')
        print(20*'*', 'CONTROL ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Perception'] == False:
        print(20*'*', 'PERCEPTION STOPPED', 20*'*')
        DREAMVIEW.enable_module('Perception')
        print(20*'*', 'PERCEPTION ENABLED', 20*'*')
        stop = True
        stop_time = max(300, stop_time)
        
    if modules_status['Routing'] == False:
        print(20*'*', 'ROUTING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Routing')
        print(20*'*', 'ROUTING ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if modules_status['Planning'] == False:
        print(20*'*', 'PLANNING STOPPED', 20*'*')
        DREAMVIEW.enable_module('Planning')
        print(20*'*', 'PLANNING ENABLED', 20*'*')
        stop = True
        stop_time = max(120, stop_time)
        
    if stop:
        time.sleep(stop_time)

def cal_dis(x_a, y_a, z_a, x_b, y_b, z_b):
    return math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)

def extract_lane_info(local_info, control_info):
    
    global current_lane
    
    for lane in control_info["lane_arr"]:
        lane_id = control_info["lane_arr"][lane]

        if ((lanes_map[lane_id]['central_curve'][0]['x'] <= local_info['position']['x'] and local_info['position']['x'] <= lanes_map[lane_id]['central_curve'][-1]['x'])
                or (lanes_map[lane_id]['central_curve'][0]['x'] >= local_info['position']['x'] and local_info['position']['x'] >= lanes_map[lane_id]['central_curve'][-1]['x'])):
            if ((lanes_map[lane_id]['central_curve'][0]['y'] <= local_info['position']['y'] and local_info['position']['y'] <= lanes_map[lane_id]['central_curve'][-1]['y'])
                    or (lanes_map[lane_id]['central_curve'][0]['y'] >= local_info['position']['y'] and local_info['position']['y'] >= lanes_map[lane_id]['central_curve'][-1]['y'])):
                
                left_bound_1 = sim.map_from_gps(None, None, lanes_map[lane_id]['left_boundary'][0]['y'], lanes_map[lane_id]['left_boundary'][0]['x'],  None, None)
                left_bound_2 = sim.map_from_gps(None, None, lanes_map[lane_id]['left_boundary'][-1]['y'], lanes_map[lane_id]['left_boundary'][-1]['x'],  None, None)
                
                left_a = left_bound_2.position.z - left_bound_1.position.z
                left_b = left_bound_1.position.x - left_bound_2.position.x
                left_c = - left_a * left_bound_1.position.x - left_b * left_bound_1.position.z

                right_bound_1 = sim.map_from_gps(None, None, lanes_map[lane_id]['right_boundary'][0]['y'], lanes_map[lane_id]['right_boundary'][0]['x'],  None, None)
                right_bound_2 = sim.map_from_gps(None, None, lanes_map[lane_id]['right_boundary'][-1]['y'], lanes_map[lane_id]['right_boundary'][-1]['x'],  None, None)
                
                right_a = right_bound_2.position.z - right_bound_1.position.z
                right_b = right_bound_1.position.x - right_bound_2.position.x
                right_c = - right_a * right_bound_1.position.x - right_b * right_bound_1.position.z

                current_lane['left_boundary']['a'] = left_a
                current_lane['left_boundary']['b'] = left_b
                current_lane['left_boundary']['c'] = left_c

                current_lane['right_boundary']['a'] = right_a
                current_lane['right_boundary']['b'] = right_b
                current_lane['right_boundary']['c'] = right_c
                
                current_lane['left_boundary_type'] = lanes_map[lane_id]['left_boundary_type']
                current_lane['right_boundary_type'] = lanes_map[lane_id]['right_boundary_type']
                
                if lanes_map[lane_id]['left_lane_direction'] < 0:
                    current_lane['left_lane_direction'] = 2
                else:
                    current_lane['left_lane_direction'] = lanes_map[lane_id]['left_lane_direction']
                    
                if lanes_map[lane_id]['right_lane_direction'] < 0:
                    current_lane['right_lane_direction'] = 2
                else:
                    current_lane['right_lane_direction'] = lanes_map[lane_id]['right_lane_direction']
                
                current_lane['vector'] = {
                    'x': lanes_map[lane_id]['central_curve'][1]['x'] - lanes_map[lane_id]['central_curve'][0]['x'],
                    'y': lanes_map[lane_id]['central_curve'][1]['y'] - lanes_map[lane_id]['central_curve'][0]['y']
                }  
                current_lane['id'] = lane_id

def is_in_junction(p_point):
    
    global junctions_map
    
    p_point = Point(p_point)
    closet_dis = 999999
    res_id = 0
    
    for id, lane_polygon in junctions_map.items():
        if lane_polygon.contains(p_point):
            return id, 2
        
        dis = lane_polygon.boundary.distance(p_point)
        if dis < closet_dis:
            closet_dis = dis
            res_id = id

    if closet_dis <= 20:
        return res_id, 1
    else:
        return None, 0
    
def cal_angle_ox(vector_):
    
    if vector_['x'] == 0:
        if vector_['y'] > 0:
            return math.pi
        else:
            return -math.pi
    
    angle = math.atan2(vector_['y'], vector_['x'])
    
    return angle
    
def entry_detail(current_lane, target_lane, dis_vector):
    up = current_lane['x'] * target_lane['x'] + current_lane['y'] * target_lane['y']
    down = math.sqrt(current_lane['x'] ** 2 + current_lane['y'] ** 2) * math.sqrt(target_lane['x'] ** 2 + target_lane['y'] ** 2) + 0.001
    
    angle = math.acos(abs(up) / down)
    
    if angle < math.pi / 12:
        if up < 0:
            return "vertical", "inverse"
        else:
            return "vertical", "forward"
    else:
        angle1 = cal_angle_ox(current_lane)
        angle2 = cal_angle_ox(dis_vector)

        if angle1 > 0:
            if angle2 >= angle1 - math.pi and angle2 <= angle1:
                return "horizontal", "right"
            else:
                return "horizontal", "left"
        else:
            if angle2 >= angle1 and angle2 <= angle1 + math.pi:
                return "horizontal", "left"
            else:
                return "horizontal", "right"

@app.route(f'{api_prefix}/status/environment/state', methods=['GET'])
def get_environment_state():

    global current_lane
    global time_offset

    agents = sim.get_agents()

    weather = sim.weather
    position = agents[0].state.position
    velocity = agents[0].state.velocity
    rotation = agents[0].state.rotation
    signal = sim.get_controllable(position, "signal")
    speed = agents[0].state.speed
    
    # calculate advanced external features
    num_obs = len(agents) - 1

    min_obs_dist = 100
    speed_min_obs_dist = 100

    for j in range(1, num_obs + 1):
        state_ = agents[j].state

        dis_to_ego = cal_dis(position.x,
                             position.y,
                             position.z,
                             state_.position.x,
                             state_.position.y,
                             state_.position.z)

        if dis_to_ego < min_obs_dist:
            min_obs_dist = dis_to_ego
            speed_min_obs_dist = state_.speed

    # get apollo info
    local_info, per_info, control_info = get_apollo_msg()
    print("Get messages")

    # extract lane info 
    extract_lane_info(local_info, control_info)
    lane_info = 0
    
    lane_info = lane_info + 100 * current_lane['left_boundary_type']
    lane_info = lane_info + 10 * current_lane['right_boundary_type']
    lane_info = lane_info + 3 * current_lane['left_lane_direction']          
    lane_info = lane_info + current_lane['right_lane_direction'] 

    # transform ego's position to world coordinate position
    transform = lgsvl.Transform(
        lgsvl.Vector(position.x, position.y,
                     position.z), lgsvl.Vector(rotation.x, rotation.y, rotation.z)
    )
    gps = sim.map_to_gps(transform)
    dest_x = gps.easting
    dest_y = gps.northing

    transform_next = lgsvl.Transform(
        lgsvl.Vector(position.x + velocity.x, position.y + velocity.y,
                     position.z + velocity.z), lgsvl.Vector(rotation.x, rotation.y, rotation.z)
    )
    gps_next = sim.map_to_gps(transform_next)
    dest_x_next = gps_next.easting
    dest_y_next = gps_next.northing

    ego_direction = {
        'x': dest_x_next - dest_x,
        'y': dest_y_next - dest_y
    }
    
    # get junction info
    junction_id, junction_position = is_in_junction([dest_x, dest_y])
    
    has_vertical_entry_inverse = False
    has_vertical_entry_forward = False
    has_horizontal_entry_left = False
    has_horizontal_entry_right = False
    
    if junction_id is not None:
        for lane_id in lanes_junctions_map[junction_id]:
            lane_vector = {
                'x': lanes_map[lane_id]['central_curve'][1]['x'] - lanes_map[lane_id]['central_curve'][0]['x'],
                'y': lanes_map[lane_id]['central_curve'][1]['y'] - lanes_map[lane_id]['central_curve'][0]['y']
            } 

            dis_vector = {
                'x': lanes_map[lane_id]['central_curve'][0]['x'] - dest_x,
                'y': lanes_map[lane_id]['central_curve'][0]['y'] - dest_y
            }    
            
            _, direct = entry_detail(ego_direction, lane_vector, dis_vector)
            
            if direct == 'left':
                has_horizontal_entry_left = True
            elif direct == 'right':
                has_horizontal_entry_right = True
            elif direct == 'inverse':
                has_vertical_entry_inverse = True
            else:
                has_vertical_entry_forward = True
                
    junction_info = 0
    
    junction_info = junction_position * 100
    
    if has_horizontal_entry_left:
        junction_info += 1
        
    if has_horizontal_entry_right:
        junction_info += 2
    
    if has_vertical_entry_forward:
        junction_info += 4
        
    if has_vertical_entry_inverse:
        junction_info += 8

    # Calculate the differences between localization and simulator

    vector_avut = np.array([
        dest_x,
        dest_y,
    ])

    vector_local = np.array([
        local_info['position']['x'],
        local_info['position']['y'],
    ])

    local_diff = np.linalg.norm(vector_local - vector_avut)

    # Calculate the angle of lcoalization's position and simulator ego's position

    v_x = vector_local[0] - vector_avut[0]
    v_y = vector_local[1] - vector_avut[1]

    local_angle = math.atan2(v_y, v_x)

    if v_x < 0:
        local_angle += math.pi

    weather_state = weather.rain * 10 ** 2 + weather.fog * 10 + weather.wetness

    state_dict = {'x': position.x, 'y': position.y, 'z': position.z,
                  'rx': rotation.x, 'ry': rotation.y, 'rz': rotation.z,
                  'weather': weather_state, 'timeofday': (sim.time_of_day - time_offset + 24) % 24, 
                  'signal': interpreter_signal(signal.current_state),
                  'speed': speed, 'local_diff': local_diff, 'local_angle': local_angle,
                  'dis_diff': per_info["dis_diff"], 'theta_diff': per_info["theta_diff"],
                  'vel_diff': per_info["vel_diff"], 'size_diff': per_info["size_diff"],
                  'throttle': control_info['throttle'], 'brake': control_info['brake'], 
                  'steering_rate': control_info['steering_rate'], 'steering_target': control_info['steering_target'], 
                  'acceleration': control_info['acceleration'], "num_obs": num_obs, 
                  "min_obs_dist": min_obs_dist, "speed_min_obs_dist": speed_min_obs_dist,
                  'lane_info': lane_info, 'junction_info': junction_info
                  }

    return json.dumps(state_dict)


@app.route(f'{api_prefix}/status/environment/weather', methods=['GET'])
def get_weather():
    weather = sim.weather
    weather_dict = {'rain': weather.rain,
                    'fog': weather.fog, 'wetness': weather.wetness}

    return json.dumps(weather_dict)


@app.route(f'{api_prefix}/status/environment/weather/rain', methods=['GET'])
def get_rain():
    return str(sim.weather.rain)


@app.route(f'{api_prefix}/status/environment/time-of-day', methods=['GET'])
def get_timeofday():
    return str(sim.time_of_day)


@app.route(f'{api_prefix}/status/collision-info', methods=['GET'])
def get_loc():

    global collision_object
    
    collision_info = str(collision_object)
    collision_object = None
    collision_type = str(None)
    
    if collision_info == 'OBSTACLE':
        collision_type = "obstacle"
    if collision_info in npc_vehicle:
        collision_type = "npc_vehicle"
    if collision_info in pedestrian:
        collision_type = "pedestrian"

    return collision_type

@app.route(f'{api_prefix}/status/collision-object', methods=['GET'])
def get_collision_object():
    global collision_object
    return str(collision_object)


@app.route(f'{api_prefix}/status/collision-uid', methods=['GET'])
def get_collision_uid():
    global collision_uid

    col_uid = collision_uid
    collision_uid = "No Collision"

    return str(col_uid)

@app.route(f'{api_prefix}/status/ego-vehicle/speed', methods=['GET'])
def get_speed():
    speed = "{:.2f}".format(sim.get_agents()[0].state.speed)
    return speed


@app.route(f'{api_prefix}/status/ego-vehicle/position', methods=['GET'])
def get_position():
    position = sim.get_agents()[0].state.position
    pos_dict = {'x': position.x, 'y': position.y, 'z': position.z}
    return json.dumps(pos_dict)

@app.route(f'{api_prefix}/status/collision-probability', methods=['GET'])
def get_c_probability():
    global probability
    c_probability = probability
    probability = 0
    return str(c_probability)

@app.route(f'{api_prefix}/status/distance-to-obstacles', methods=['GET'])
def get_distance_to_obstacles():
    global DTO
    dto = DTO
    DTO = 100000
    return str(dto)


@app.route(f'{api_prefix}/status/estimated-time-to-collision', methods=['GET'])
def get_estimated_time_to_collision():
    global ETTC
    ettc = ETTC
    ETTC = 100
    return str(ettc)

@app.route(f'{api_prefix}/status/jerk', methods=['GET'])
def get_jerk():
    global JERK
    jerk = JERK
    JERK = 100
    return str(jerk)

if __name__ == '__main__':
    connect_to_apollo_listener()
    load_map_traffic_condition()
    app.run(host=API_SERVER_HOST, port=API_SERVER_PORT, debug=False)