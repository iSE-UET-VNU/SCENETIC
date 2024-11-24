import math
import numpy as np
import random
from lgsvl.agent import NpcVehicle
from numba import jit

print("Using collision utils")

pedestrian = ["Bob", "EntrepreneurFemale", "Howard", "Johny", "Pamela", "Presley", "Robin", "Stephen", "Zoe"]

npc_vehicle = {"Sedan", "SUV", "Jeep", "Hatchback", "SchoolBus", "BoxTruck"}

@jit(nopython=True, fastmath=True)
def calculate_angle_tan(k1, k2):
    if k1 == k2:
        k2 = k2 - 0.0001
    tan_theta = abs((k1 - k2) / (1 + k1 * k2))
    theta = np.arctan(tan_theta)
    return theta

@jit(nopython=True, fastmath=True)
def calculate_angle(vector1, vector2):
    cos_theta = (vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]) / \
                (math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2]) *
                  math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1] + vector2[2] * vector2[2]))
    theta = np.arccos(cos_theta)

    return theta

def calculate_distance(vector1, vector2):
    distance = math.sqrt(pow(vector1[0] - vector2[0], 2) + pow(vector1[1] - vector2[1], 2) + pow(vector1[2] - vector2[2], 2))
    return distance

# Calculate collision probability
@jit(nopython=True, fastmath=True)
def calculate_collision_probability(safe_distance, current_distance):
    collision_probability = None
    if current_distance >= safe_distance:
        collision_probability = 0
    elif current_distance < safe_distance:
        collision_probability = (safe_distance - current_distance) / safe_distance
    return collision_probability
    
    
@jit(nopython=True, fastmath=True)
def get_line(agent_position, agent_velocity):
    agent_position_x = agent_position[0]
    agent_position_z = agent_position[2]

    agent_velocity_x = agent_velocity[0] if agent_velocity[0] != 0 else 0.0001
    agent_velocity_z = agent_velocity[2]

    return agent_velocity_z / agent_velocity_x, -(
                agent_velocity_z / agent_velocity_x) * agent_position_x + agent_position_z

@jit(nopython=True, fastmath=True)
def get_distance(agent_position, x, z):
    return math.sqrt(pow(agent_position[0] - x, 2) + pow(agent_position[2] - z, 2))

@jit(nopython=True, fastmath=True)
def judge_same_line(a1_position, a1_speed, a1_velocity, a2_position, a2_speed, k1, k2):
    judge = False
    ego_ahead = False
    direction_vector = (a1_position[0] - a2_position[0], a1_position[2] - a2_position[2])
    distance = get_distance(a1_position, a2_position[0], a2_position[2])

    if abs(k1 - k2) < 0.6:
        if abs((a1_position[2] - a2_position[2]) /
               ((a1_position[0] - a2_position[0]) if (a1_position[0] - a2_position[0]) != 0 else 0.01) - (k1 + k2) / 2) < 0.6:
            judge = True

    if not judge:
        return judge, ego_ahead, -1

    if direction_vector[0] * a1_velocity[0] >= 0 and direction_vector[1] * a1_velocity[2] >= 0:
        ego_ahead = True 
        TTC = distance / (a1_speed - a2_speed)
    else:
        TTC = distance / (a2_speed - a1_speed)
    if TTC < 0:
        TTC = 100000

    return judge, ego_ahead, TTC

def calculate_measures(state_list, ego_state, is_npc_vehicle):
    
    ego_transform = ego_state.transform
    ego_speed = ego_state.speed if ego_state.speed > 0 else 0.0001
    ego_position = np.array([ego_transform.position.x, ego_transform.position.y, ego_transform.position.z])
    ego_velocity = np.array([ego_state.velocity.x, ego_state.velocity.y, ego_state.velocity.z])

    trajectory_ego_k, trajectory_ego_b = get_line(ego_position, ego_velocity)

    TTC = 100000
    lo_proc_list, la_proc_list = [0], [0] 
    
    for i in range(0, len(state_list)):
        transform = state_list[i].transform
        state = state_list[i]
        a_position = np.array([transform.position.x, transform.position.y, transform.position.z])
        a_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z])
        
        distance = get_distance(ego_position, a_position[0], a_position[2])
    
        trajectory_agent_k, trajectory_agent_b = get_line(a_position, a_velocity)
        
        agent_speed = 0.0001
        
        if is_npc_vehicle[i]:
            agent_speed = state.speed if state.speed > 0 else 0.0001
        else:
            agent_speed = math.sqrt(a_velocity[0] ** 2 + a_velocity[1] ** 2 + a_velocity[2] ** 2)
            agent_speed = max(agent_speed, 0.0001)

        same_lane, _, ttc = judge_same_line(ego_position, ego_speed, ego_velocity,
                                                    a_position, state.speed, trajectory_ego_k, trajectory_agent_k)
        ego_deceleration = 6  
        if same_lane:
            
            time_ego = ttc
            time_agent = ttc

            lo_sd = 100000
            if is_npc_vehicle[i]: 
                agent_deceleration = 6
                lo_sd = 1 / 2 * (
                    abs(pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration)) + 5
            else:
                agent_deceleration = 1.5
                lo_sd = 1 / 2 * abs(pow(ego_speed, 2) / ego_deceleration - pow(agent_speed, 2) / agent_deceleration) + 5
            lo_proc = calculate_collision_probability(lo_sd, distance)
            lo_proc_list.append(lo_proc)

        else:
            trajectory_agent_k = trajectory_agent_k if trajectory_ego_k - trajectory_agent_k != 0 else trajectory_agent_k + 0.0001

            collision_point_x, collision_point_z = (trajectory_agent_b - trajectory_ego_b) / (
                        trajectory_ego_k - trajectory_agent_k), \
                                                   (
                                                               trajectory_ego_k * trajectory_agent_b - trajectory_agent_k * trajectory_ego_b) / (
                                                               trajectory_ego_k - trajectory_agent_k)

            ego_distance = get_distance(ego_position, collision_point_x, collision_point_z)
            agent_distance = get_distance(a_position, collision_point_x, collision_point_z)
            time_ego = ego_distance / ego_speed
            time_agent = agent_distance / agent_speed

            theta = calculate_angle_tan(trajectory_ego_k, trajectory_agent_k)
   
            la_sd = pow(ego_speed * math.sin(theta), 2) / (ego_deceleration * math.sin(theta)) + 5
            la_proc = calculate_collision_probability(la_sd, distance)
            la_proc_list.append(la_proc)

           

        if abs(time_ego - time_agent) < 1:
            TTC = min(TTC, (time_ego + time_agent) / 2)

    lo_proc_dt, la_proc_dt = max(lo_proc_list), max(la_proc_list)
    proc_dt = max(lo_proc_dt, la_proc_dt) + (1 - max(lo_proc_dt, la_proc_dt)) * min(lo_proc_dt, la_proc_dt)

    return TTC, 5, proc_dt
