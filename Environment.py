import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import jv
import heapq
import random
import Unit as ut
import pandas as pd
import parameters as pa

class env():
    def __init__(self):
        # === [추가] Outdated CSI: 이전 step 채널 저장 (beam × user) ===
        self.h_prev = np.ones((pa.number_of_beam, pa.number_of_user), dtype=complex) * 1e-10
        return None

    # ---------------------------------------Network Architecture-----------------------------------------
    def initial_user_location(self):
        user_x = np.zeros((pa.number_of_satellite, pa.number_of_user))
        user_y = np.zeros((pa.number_of_satellite, pa.number_of_user))
        user_z = np.zeros((pa.number_of_satellite, pa.number_of_user))
        user_location_phi = np.zeros((pa.number_of_satellite, pa.number_of_user))
        user_location_theta = np.zeros((pa.number_of_satellite, pa.number_of_user))

        for net in range(pa.number_of_area):
            phi_min = pa.Nphi[int((pa.Round/2)-450)]
            phi_max = pa.Nphi[int((pa.Round/2)+450)]
            theta_min = pa.Ntheta[int(pa.number_of_iteration*net)]
            theta_max = pa.Ntheta[int((pa.number_of_iteration*(net+1)))-1]
            phi_center = (phi_min + phi_max) / 2
            theta_center = (theta_min + theta_max) / 2
            net_t = pa.net_type[net]

            for user in range(pa.number_of_user):
                if net_t == 0:
                    phi = np.clip(np.random.normal(phi_center, pa.urban_sigma_phi), phi_min, phi_max)
                    theta = np.clip(np.random.normal(theta_center, pa.urban_sigma_theta), theta_min, theta_max)
                else:  
                    phi = random.uniform(phi_min, phi_max)
                    theta = random.uniform(theta_min, theta_max)

                user_location_phi[net][user] = phi
                user_location_theta[net][user] = theta

        for SAT in range(pa.number_of_satellite):
            for user in range(pa.number_of_user):
                user_x[SAT][user] = pa.radius * math.sin(user_location_phi[SAT][user]) * math.cos(user_location_theta[SAT][user])
                user_y[SAT][user] = pa.radius * math.sin(user_location_phi[SAT][user]) * math.sin(user_location_theta[SAT][user])
                user_z[SAT][user] = pa.radius * math.cos(user_location_phi[SAT][user])

        return user_x, user_y, user_z, user_location_phi, user_location_theta


    def User_requirment(self,episode):
        rng = np.random.default_rng(seed=0)
        result = np.zeros((pa.number_of_satellite, pa.number_of_user), dtype=int)
        for net in range(pa.number_of_satellite):
            ratio = pa.active_ratio[pa.net_type[net]]
            n_active = int(pa.number_of_user * ratio)
            active_indices = rng.choice(pa.number_of_user, n_active, replace=False)
            result[net][active_indices] = 1
        return result

    def move_to_user(self, user_x, user_y, user_z, user_location_phi, user_location_theta, network):
        dphi = np.random.uniform(-pa.move_radius / pa.radius, pa.move_radius / pa.radius, pa.number_of_user)
        dtheta = np.random.uniform(-pa.move_radius / (pa.radius * np.sin(user_location_phi[network])), pa.move_radius / (pa.radius * np.sin(user_location_phi[network])),pa.number_of_user)
        new_phi = np.clip(user_location_phi[network] + dphi, pa.Nphi[int((pa.Round/2)-450)], pa.Nphi[int((pa.Round/2)+450)])
        new_theta = np.clip(user_location_theta[network] + dtheta,pa.Ntheta[int(pa.number_of_iteration*network)], pa.Ntheta[int((pa.number_of_iteration*(network+1)))-1])
        user_location_phi[network] = new_phi
        user_location_theta[network] = new_theta
        new_user_x = pa.radius * np.sin(new_phi) * np.cos(new_theta)
        new_user_y = pa.radius * np.sin(new_phi) * np.sin(new_theta)
        new_user_z = pa.radius * np.cos(new_phi)
        user_x[network] = new_user_x
        user_y[network] = new_user_y
        user_z[network] = new_user_z
        return user_x, user_y, user_z, user_location_phi, user_location_theta


    def MultiBeam_create(self):
        if pa.number_of_beam == 6:
            h = [[-6,3],[0,0],[6,3],[6,12],[0,9],[-6,12]]
        elif pa.number_of_beam == 8:
            h = [[4,2],[0,5],[-4,2],[-4,8],[4,8],[4,14],[0,11],[-4,14]]

        else:
            h = [[-6,3],[-3,0],[0,3],[3,0],[6,3],[3,6],[6,9],[3,12],[0,9],[-3,12],[-6,9],[-3,6]]

        Beam_x = np.zeros((pa.number_of_satellite, pa.number_of_beam))
        Beam_y = np.zeros((pa.number_of_satellite, pa.number_of_beam))
        Beam_z = np.zeros((pa.number_of_satellite, pa.number_of_beam))
        for SAT in range(pa.number_of_satellite):
            for Beam, odd in enumerate(h):
                Beam_x[SAT][Beam] = pa.radius * math.sin(pa.Nphi[int((pa.Round/2)+(odd[0]*int(pa.number_of_iteration/16)))]) \
                                    * math.cos(pa.Ntheta[int((pa.number_of_iteration*SAT)+(odd[1]*int(pa.number_of_iteration/16)))])
                Beam_y[SAT][Beam] = pa.radius * math.sin(pa.Nphi[int((pa.Round/2)+(odd[0]*int(pa.number_of_iteration/16)))]) \
                                    * math.sin(pa.Ntheta[int((pa.number_of_iteration*SAT)+(odd[1]*int(pa.number_of_iteration/16)))])
                Beam_z[SAT][Beam] = pa.radius * math.cos(pa.Nphi[int((((pa.number_of_area*pa.number_of_iteration))/2)+(odd[0]*int(pa.number_of_iteration/16)))])
        return Beam_x, Beam_y, Beam_z

    def initial_satellite_location(self):
        satellite_x = np.zeros(pa.number_of_satellite)
        satellite_y = np.zeros(pa.number_of_satellite)
        satellite_z = np.zeros(pa.number_of_satellite)
        for SAT in range(pa.number_of_satellite):
            satellite_x[SAT] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round/2))]) * math.cos(pa.Ntheta[pa.number_of_iteration*SAT])
            satellite_y[SAT] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round/2))]) * math.sin(pa.Ntheta[pa.number_of_iteration*SAT])
            satellite_z[SAT] = (pa.radius + pa.altitude) * math.cos(pa.Nphi[int((pa.Round/2))])
        return satellite_x, satellite_y, satellite_z

    def move_to_satellite(self):
        step_interval_x = np.zeros((pa.number_of_satellite,pa.number_of_iteration))
        step_interval_y = np.zeros((pa.number_of_satellite,pa.number_of_iteration))
        step_interval_z = np.zeros((pa.number_of_satellite,pa.number_of_iteration))

        for SAT in range(pa.number_of_satellite):
            for timestep in range(pa.number_of_iteration):
                if SAT == pa.number_of_satellite - 1 and timestep == pa.number_of_iteration - 1:
                    step_interval_x[SAT][timestep] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round / 2))]) * math.cos(pa.Ntheta[0])
                    step_interval_y[SAT][timestep] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round / 2))]) * math.sin(pa.Ntheta[0])
                    step_interval_z[SAT][timestep] = (pa.radius + pa.altitude) * math.cos(pa.Nphi[int((pa.Round / 2))])
                else:
                    step_interval_x[SAT][timestep] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round/2))]) * math.cos(pa.Ntheta[(pa.number_of_iteration * SAT) + timestep + 1])
                    step_interval_y[SAT][timestep] = (pa.radius + pa.altitude) * math.sin(pa.Nphi[int((pa.Round/2))]) * math.sin(pa.Ntheta[(pa.number_of_iteration * SAT) + timestep + 1])
                    step_interval_z[SAT][timestep] = (pa.radius + pa.altitude) * math.cos(pa.Nphi[int((pa.Round/2))])

        return step_interval_x, step_interval_y, step_interval_z

    # ----------------------------------------------- distance 구하기 ----------------------------------------------------
    def calculate_distance_sat2user(self, user_x, user_y, user_z, satellite_x, satellite_y, satellite_z, satellite, timestep):
        distance_sat2user = np.zeros(pa.number_of_user)
        for user in range(pa.number_of_user):
            distance_sat2user[user] = math.sqrt(
                (satellite_x[satellite][timestep] - user_x[satellite][user]) ** 2 + (satellite_y[satellite][timestep] - user_y[satellite][user]) ** 2 + (
                        satellite_z[satellite][timestep] - user_z[satellite][user]) ** 2)
            if distance_sat2user[user] < pa.altitude:
                distance_sat2user[user] = pa.altitude
        return distance_sat2user

    def calculate_distance_sat2Beam(self, Beam_x, Beam_y, Beam_z, satellite_x, satellite_y, satellite_z, satellite,timestep):
        distance_sat2Beam = np.sqrt((satellite_x[satellite][timestep] - Beam_x[satellite]) ** 2 + (satellite_y[satellite][timestep] - Beam_y[satellite]) ** 2 + (satellite_z[satellite][timestep] - Beam_z[satellite]) ** 2)
        return distance_sat2Beam

    def calculate_distance_Beam2user(self, Beam_x, Beam_y, Beam_z, user_x, user_y, user_z, satellite):
        distance_Beam2user = np.zeros((pa.number_of_beam, pa.number_of_user))
        for Beam in range(pa.number_of_beam):
            distance_Beam2user[Beam] = np.sqrt((Beam_x[satellite][Beam] - user_x[satellite]) ** 2 + (Beam_y[satellite][Beam] - user_y[satellite]) ** 2 +
                                                           (Beam_z[satellite][Beam] - user_z[satellite]) ** 2)
        return distance_Beam2user

# ------------------------------------------------- angle 구하기 ----------------------------------------------------
    def calculate_elevation_angle(self, distance_sat2user):
        a = pa.radius
        b = 7600*10**3
        c = distance_sat2user
        elevation_angle = np.arccos(np.round(((a ** 2) + (c ** 2) - (b ** 2)) / (2 * a * c),5))
        elevation_angle= np.rad2deg(elevation_angle) - 90
        elevation_angle = np.round(elevation_angle, -1)
        return elevation_angle

    def calculate_angle(self, distance_sat2Beam, distance_Beam2user, distance_sat2user):
        a = distance_Beam2user
        b = distance_sat2Beam
        c = distance_sat2user
        angle_BSU = np.zeros((pa.number_of_beam, pa.number_of_user))
        angle = np.zeros((pa.number_of_beam, pa.number_of_user))
        for beam in range(pa.number_of_beam):
            angle[beam] = ((b[beam] ** 2) + (c ** 2) - (a[beam] ** 2)) / (2 * b[beam] * c)
            angle_BSU[beam] = np.arccos(angle[beam])
        return angle_BSU

    def calculate_LoS_probability(self, elevation_angle):
        LoS_probability = {10: 78.2, 20: 86.9, 30: 91.9, 40: 92.9, 50: 93.5, 60: 94.0, 70: 94.9, 80: 95.2, 90: 99.8}
        RandomChoice = np.random.uniform(0,100)
        LoS = np.zeros(pa.number_of_user)
        for user in range(pa.number_of_user):
            if elevation_angle[user] >= 10:
                if RandomChoice >= LoS_probability[elevation_angle[user]]:
                    LoS[user] = 0
                elif RandomChoice < LoS_probability[elevation_angle[user]]:
                    LoS[user] = 1
            else:
                LoS[user] = 0
        return LoS


    def calculate_pathloss(self, LoS, doppler_freq_, distance_sat2user, elevation_angle):
        SF_factor = np.zeros(pa.number_of_user)
        clutter_loss = np.zeros(pa.number_of_user)
        for user in range(pa.number_of_user):
            if LoS[user] == 1:
                SF_factor_ = {10: 1.79, 20: 1.14, 30: 1.14, 40: 0.92, 50: 1.42, 60: 1.56, 70: 0.85, 80: 0.72,
                                        90: 0.72}
                clutter_loss_ = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0}
            else:
                SF_factor_  = {10: 8.93, 20: 9.08, 30: 8.78, 40: 10.25, 50: 10.56, 60: 10.74, 70: 10.17, 80: 11.52,
                                        90: 11.52}
                clutter_loss_ = {10: 19.52, 20: 18.17, 30: 18.42, 40: 18.28, 50: 18.63, 60: 17.68, 70: 16.50, 80: 16.3,
                                90: 16.3}
            clutter_loss[user] = clutter_loss_[elevation_angle[user]]
            SF_factor[user] = SF_factor_[elevation_angle[user]]

        SF = np.random.normal(0, SF_factor, pa.number_of_user)
        FSPL = 32.45 + (20 * np.log10(doppler_freq_ / (10 ** (9)))) + (20 * np.log10(distance_sat2user))

        pathloss = FSPL + SF + clutter_loss
        return pathloss
    
    
    def smallscalefading(self, elevation_angle, LoS):
        mu_K = np.vectorize(lambda e: pa.K_FACTOR_TABLE.get(e, (12.85, 9.91))[0])(elevation_angle)
        sigma_K = np.vectorize(lambda e: pa.K_FACTOR_TABLE.get(e, (12.85, 9.91))[1])(elevation_angle)

        K_dB = np.random.normal(mu_K, sigma_K)
        K = np.vectorize(ut.dB2watt)(K_dB)
        K = np.maximum(K, 20)

        LoS_mask = (LoS == 1)[np.newaxis, :]  

        mu = np.where(LoS_mask, np.sqrt(K / (K + 1)), 0.0)
        sigma = np.where(LoS_mask, np.sqrt(1 / (K + 1)), 1.0)

        noise = (np.random.randn(*mu.shape) + 1j * np.random.randn(*mu.shape)) / np.sqrt(2)
        u = mu + sigma * noise

        h = pa.RHO_D * self.h_prev + np.sqrt(1 - pa.RHO_D ** 2) * u
        self.h_prev = h.copy()
        return h

    def calculate_channel_gain(self, angle_beam2user, pathloss, doppler_freq_):
        channel_gain = np.zeros((pa.number_of_beam, pa.number_of_user))
        Bessel_function = np.zeros((pa.number_of_beam, pa.number_of_user))
        mue = np.zeros((pa.number_of_beam, pa.number_of_user))
        tx_antenna_gain = np.zeros((pa.number_of_beam, pa.number_of_user))
        sat_TXgain_max = (pa.antenna_efficiency * 4 * np.pi * pa.antenna_area) / ((pa.velocity_of_light / doppler_freq_) ** 2)
        for beam in range(pa.number_of_beam):
            for user in range(pa.number_of_user):
                mue[beam][user] = (2.07123 * np.sin(angle_beam2user[beam][user])) / np.sin(pa.angle_3dB)
                if mue[beam][user] == 0:
                    Bessel_function[beam][user] = 1
                else:
                    Bessel_function[beam][user] = ((jv(1, mue[beam][user]) / (
                                2 * mue[beam][user])) + (36 * jv(3, mue[beam][user]) / (mue[beam][user] ** 3))) ** 2
                tx_antenna_gain[beam][user] = sat_TXgain_max * Bessel_function[beam][user]
                channel_gain[beam][user] = tx_antenna_gain[beam][user] * ut.dB2watt(pa.user_RXgain_max) / ut.dB2watt(pathloss[user])  # watt
        return channel_gain

    def calculate_RSSI(self, channel_gain,h, current_state, sat):
        RSSI = np.zeros((pa.number_of_beam, pa.number_of_user))
        for user in range(pa.number_of_user):
            for Beam in range(pa.number_of_beam):
                RSSI[Beam][user] = current_state[sat][Beam + 1] * channel_gain[Beam][user] * np.abs(h[Beam][user])**2 # watt # s번째 위성의 b번째 빔에서 u번 user로의 신호
        return RSSI

    def calculate_SINR(self, RSSI, current_state, sat):
        noise = (pa.bandwidth / pa.number_of_beam) * (ut.dB2watt(-204))
        interference_ = np.zeros((pa.number_of_beam, pa.number_of_user))
        interference = np.zeros((pa.number_of_beam, pa.number_of_user))
        SINR = np.zeros((pa.number_of_beam, pa.number_of_user))
        for beam in range(pa.number_of_beam):
            inter = int(beam % current_state[sat][0])
            for user in range(pa.number_of_user):
                interference_[inter][user] += RSSI[beam][user]
                if RSSI[beam][user] == 0.0000000000000:
                    SINR[beam][user] = ut.dB2watt(-1000000000000000000000)
                else:
                    interference[beam][user] = interference_[inter][user] - RSSI[beam][user]
                    SINR[beam][user] = RSSI[beam][user]/(interference[beam][user] + noise) # watt
        return SINR

    def channel_allocation(self, SINR, current_state,user_traffic, sat):
        sub_channel = int(pa.number_of_channel / current_state[sat][0])
        ChannelAllocation = np.zeros((pa.number_of_beam, sub_channel, pa.number_of_user))
        Sequence_SINR = np.sort(SINR.reshape(pa.number_of_beam * pa.number_of_user), axis=0)
        able = sub_channel * np.ones(pa.number_of_beam)
        for beam in range(pa.number_of_beam):
            if current_state[sat][beam+1] == 0:
                able[beam] = 0
        for index in (Sequence_SINR[::-1]):
            try_access = np.argwhere(SINR == index)
            for i in range(np.shape(try_access)[0]):
                if able[try_access[i][0]] > 0 and np.all(ChannelAllocation[:,:,try_access[i][1]] == 0) and user_traffic[try_access[i][1]]==1:
                    ChannelAllocation[try_access[i][0]][int(able[try_access[i][0]]-1)][try_access[i][1]] = 1
                    able[try_access[i][0]] -= 1
            if np.all(able == 0):
                break
            if np.all(np.sum(ChannelAllocation, axis = (0,1)) == user_traffic):
                break
        return ChannelAllocation

    def channel_allocation_RR(self, SINR, current_state, user_traffic, sat):
        sub_channel = int(pa.number_of_channel / current_state[sat][0])
        ChannelAllocation = np.zeros((pa.number_of_beam, sub_channel, pa.number_of_user))
        able = sub_channel * np.ones(pa.number_of_beam)
        for beam in range(pa.number_of_beam):
            if current_state[sat][beam + 1] == 0:
                able[beam] = 0

        active_users = [u for u in range(pa.number_of_user) if user_traffic[u] == 1]

        for user in active_users:
            beam_sinr = [(SINR[b][user], b) for b in range(pa.number_of_beam) if able[b] > 0]
            if not beam_sinr:
                continue
            _, best_beam = max(beam_sinr)
            ChannelAllocation[best_beam][int(able[best_beam] - 1)][user] = 1
            able[best_beam] -= 1
            if np.all(able == 0):
                break

        return ChannelAllocation

    def channel_allocation_PF(self, SINR, current_state, user_traffic, sat, avg_rate):
        sub_channel = int(pa.number_of_channel / current_state[sat][0])
        ChannelAllocation = np.zeros((pa.number_of_beam, sub_channel, pa.number_of_user))
        able = sub_channel * np.ones(pa.number_of_beam)
        for beam in range(pa.number_of_beam):
            if current_state[sat][beam + 1] == 0:
                able[beam] = 0

        bw_per_ch = pa.bandwidth / pa.number_of_channel
        inst_rate = bw_per_ch * np.log2(1 + np.maximum(SINR, 1e-10))  # (beam, user)

        avg_safe = np.where(avg_rate < 1e-10, 1e-10, avg_rate)  # (user,)
        pf_metric = inst_rate / avg_safe[np.newaxis, :]  # (beam, user)

        for u in range(pa.number_of_user):
            if user_traffic[u] == 0:
                pf_metric[:, u] = -np.inf

        user_assigned = np.zeros(pa.number_of_user, dtype=bool)
        flat_indices = np.argsort(pf_metric.ravel())[::-1]

        for flat_idx in flat_indices:
            beam = flat_idx // pa.number_of_user
            user = flat_idx % pa.number_of_user
            if able[beam] > 0 and not user_assigned[user] and user_traffic[user] == 1:
                ChannelAllocation[beam][int(able[beam] - 1)][user] = 1
                able[beam] -= 1
                user_assigned[user] = True
            if np.all(able == 0):
                break

        return ChannelAllocation

    def initial_state(self, state_size,satellite_x, satellite_y, satellite_z):
        current_state = []
        Beam_power = np.ones(pa.number_of_beam) * pa.kind_of_power[2]
        frequency_reuse = pa.kind_of_frequency_reuse[1]
        for sat in range(pa.number_of_satellite):
            current_state = np.append(current_state, np.hstack((frequency_reuse, Beam_power, satellite_x[sat], satellite_y[sat], satellite_z[sat])))
        current_state = current_state.reshape((pa.number_of_satellite, state_size))
        return current_state

    def set_MDP_state(self, current_state, sat):
        state_input = current_state.copy()
        state_input[sat][0] = current_state[sat][0] / pa.number_of_beam
        state_input[sat][1:pa.number_of_beam+1] = current_state[sat][1:pa.number_of_beam+1] / pa.max_transmission_power
        state_input[sat][-3] = (state_input[sat][-3] + pa.satellite_max_location)/(2 * pa.satellite_max_location)
        state_input[sat][-2] = (state_input[sat][-2] + pa.satellite_max_location)/(2 * pa.satellite_max_location)
        state_input[sat][-1] = (state_input[sat][-1] + pa.satellite_max_location)/(2 * pa.satellite_max_location)
        state_input[sat] = np.array(state_input[sat], ndmin=2)
        return state_input

    def set_MDP_action(self, current_state, action,move_to_sat_x, move_to_sat_y, move_to_sat_z, sat, timestep):  # 주파수 재사용 계수만 조절하는 코드 (+,-,stay)
        next_state = current_state.copy()
        if action == 0: # 주파수 재사용 계수가 1이 아니면 감소
            if current_state[sat][0] == 1:
                next_state[sat][0] = current_state[sat][0]
            else:
                new_index = pa.kind_of_frequency_reuse.index(current_state[sat][0]) - 1
                next_state[sat][0] = pa.kind_of_frequency_reuse[new_index]
        elif action == 1: # 주파수 재사용 계수가 8이 아니라면 증가
            if current_state[sat][0] == pa.kind_of_frequency_reuse[-1] :
                next_state[sat][0] = current_state[sat][0]
            else:
                new_index = pa.kind_of_frequency_reuse.index(current_state[sat][0]) + 1
                next_state[sat][0] = pa.kind_of_frequency_reuse[new_index]

        elif action % 2 == 0 and action != (pa.number_of_beam + 1) * 2 and action !=0: # 해당 빔의 전력이 0이 아니라면 전력 감소
            if current_state[sat][int(action / 2)] == 0:
                next_state[sat][int(action / 2)] = current_state[sat][int(action / 2)]
            else:
                new_index = pa.kind_of_power.index(current_state[sat][int(action / 2)]) - 1
                next_state[sat][int(action / 2)] = pa.kind_of_power[new_index]

        elif action % 2 == 1 and action != 1 : # 해당 빔의 전력이 25가 아니라면 전력 증가
            if current_state[sat][int(action / 2)] == pa.max_transmission_power:  # w
                next_state[sat][int(action / 2)] = current_state[sat][int(action / 2)]
            else:
                new_index = pa.kind_of_power.index(current_state[sat][int(action / 2)]) + 1
                next_state[sat][int(action / 2)] = pa.kind_of_power[new_index]

        elif action == (pa.number_of_beam + 1) * 2 :
            next_state[sat] = current_state[sat]

        next_state[sat][-3] = move_to_sat_x[sat][timestep]
        next_state[sat][-2] = move_to_sat_y[sat][timestep]
        next_state[sat][-1] = move_to_sat_z[sat][timestep]

        next_state_input = next_state.copy()
        next_state_input[sat][0] = next_state_input[sat][0] / pa.number_of_beam
        next_state_input[sat][1:pa.number_of_beam + 1] = next_state_input[sat][1:pa.number_of_beam+1]  / pa.max_transmission_power
        next_state_input[sat][-3] = (next_state_input[sat][-3] + pa.satellite_max_location) / (2 * pa.satellite_max_location)
        next_state_input[sat][-2] = (next_state_input[sat][-2] + pa.satellite_max_location) / (2 * pa.satellite_max_location)
        next_state_input[sat][-1] = (next_state_input[sat][-1] + pa.satellite_max_location) / (2 * pa.satellite_max_location)

        return next_state, next_state_input

    # === [수정] calculate_reward: ISL 전송 비용 포함 (Reviewer 1.2) ===
    def calculate_reward(self, ChannelAllocation, SINR, current_state, sat, method):
        ADR = np.zeros((pa.number_of_beam, pa.number_of_user))
        power_consumptions = np.zeros(pa.number_of_beam)
        for Beam in range(pa.number_of_beam):
            power_consumptions[Beam] = np.sum(ChannelAllocation[Beam]) * current_state[sat][Beam+1]
            for user in range(pa.number_of_user):
                if SINR[Beam][user] >= 0.0000000000000000000:
                    ADR[Beam][user] = (pa.bandwidth / pa.number_of_channel ) * np.log2(1 + SINR[Beam][user]) * sum(ChannelAllocation[Beam,:,user])

        if sum(current_state[sat][1:pa.number_of_beam+1]) == 0:
            reward = 0
        else:
            # === [추가] ISL 전송 비용 포함 (Reviewer 1.2) ===
            P_isl = pa.P_ISL.get(method, 0.0)
            EE = np.sum(ADR) / (np.sum(power_consumptions) + P_isl)
            reward = EE
        Normalization_reward = reward / 10000000
        return ADR,reward, Normalization_reward
