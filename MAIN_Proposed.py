from datetime import datetime, timedelta
from collections import deque
from Environment import env
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import Unit as ut
import parameters as pa
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import legacy # 파이썬 버전에 따라서 주석 조절.
from tensorflow.keras import optimizers # 파이썬 버전에 따라서 주석 조절.

class DQN:
    def __init__(self, state_size, action_size):
        # ------------------------사용자 조절 파라미터 ------------------------
        self.number_of_agent = 20  # agent 수
        self.number_of_cycle = 1000  # episode 수
        self.learning_rate = 0.00005 # 학습률
        self.epsilon = 0.99 # 초기 입실론 값
        self.epsilon_decay = 0.994 # 입실론 decay 값
        self.epsilon_min = 0 # 입실론 최소값
        self.batch_size = 64 # 배치 사이즈
        self.discount_factor = 0.95 # 가감률
        self.train_start = 200  # 학습 시작
        self.node = 64 # 딥러닝 뉴런 수
        self.target_update_period = 150
        # ------------------------------------------------------------------
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = {}
        # self.optimizer = optimizers.Adam(learning_rate = self.learning_rate) # 파이썬 버전에 따라서 주석 조절.
        # self.optimizer = legacy.Adam(learning_rate=self.learning_rate)  # 파이썬 버전에 따라서 주석 조절.
        # ------------------------------------------------------------------------------------------------
        for sat in range(self.number_of_agent):
            globals()['model{}'.format(sat)] = self.Create_Model(state_size, action_size)
            globals()['target_model{}'.format(sat)] = self.Create_Model(state_size, action_size)
            globals()['memory{}'.format(sat)] = deque(maxlen=10000)
            globals()['score_sat{}'.format(sat)] = []
            globals()['score_net{}'.format(sat)] = []
            self.optimizer[sat] = optimizers.Adam(learning_rate=self.learning_rate)


#------------------------------------------------------DQN Training Code---------------------------------------------------------------------------

    def Create_Model(self, state_size, action_size):
        inputs_layer = tf.keras.Input(shape=(state_size,))
        hidden_layer1 = tf.keras.layers.Dense(self.node, activation=tf.nn.relu)(inputs_layer)
        hidden_layer2 = tf.keras.layers.Dense(self.node, activation=tf.nn.relu)(hidden_layer1)
        output_layer = tf.keras.layers.Dense(action_size)(hidden_layer2)
        model = tf.keras.Model(inputs_layer, output_layer)
        return model

    def get_action(self, state, network):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(globals()['model{}'.format(network)](np.array(state[network], ndmin=2)))
        return action

    def append_sample(self, state, action, reward, next_state, network):
        state_ = np.array(state[network], ndmin=2)
        next_state_ = np.array(next_state[network], ndmin=2)
        globals()['memory{}'.format(network)].append((state_, action, reward, next_state_))

    def train_model(self, network):
        mini_batch = random.sample(globals()['memory{}'.format(network)], self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards = [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
        model_params = globals()['model{}'.format(network)].trainable_variables

        with tf.GradientTape() as tape:
            predicts = globals()['model{}'.format(network)](states)
            one_hot_action = tf.one_hot(np.array(actions), self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = globals()['target_model{}'.format(network)](next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        grads = tape.gradient(loss, model_params)
        # self.optimizer.apply_gradients(zip(grads, model_params))
        self.optimizer[network].apply_gradients(zip(grads, model_params))
        return states, actions, rewards, next_states

    def update_target_model(self, network):
        globals()['target_model{}'.format(network)].set_weights(globals()['model{}'.format(network)].get_weights())

    # ----------------------------------------------------------------------------------------------------------------------------
    def handover(self, cycle, sat):
        step_to_blank = cycle % self.number_of_agent
        satellite = sat
        if sat + step_to_blank < (self.number_of_agent - 1):
            satellite = sat + step_to_blank
        elif sat + step_to_blank > (self.number_of_agent - 1):
            satellite = (sat + step_to_blank - self.number_of_agent)
        elif sat + step_to_blank == (self.number_of_agent - 1):
            satellite = self.number_of_agent - 1
        return satellite
    #----------------------------------------------------------------------------------------------------------------------------




if __name__ == "__main__":
    # === Keras 버전 자동 감지 (로컬/서버 호환) ===
    import os
    os.makedirs('./save_weights', exist_ok=True)
    _keras_major = int(tf.keras.__version__.split('.')[0])
    _weight_ext = '.weights.h5' if _keras_major >= 3 else '.h5'
    # ==============================================
    env = env()
    if pa.number_of_beam == 6:
        state_size = 10
        action_size = 15
    elif pa.number_of_beam == 8:
        state_size = 12
        action_size = 19
    else :
        state_size = 16
        action_size = 27
    agent = DQN(state_size, action_size)

# # ------------------------------------사용자 위치 정보 저장 코드 -----------------------------------------------
#     user_x, user_y, user_z, user_location_phi, user_location_theta= env.initial_user_location()
#     df_user_location = {"user_x": user_x, "user_y": user_y, "user_z": user_z}
#     df_vector = {"user_location_phi":user_location_phi,"user_location_theta": user_location_theta}
#     for sat in range(pa.number_of_satellite):
#         globals()['user_location{}'.format(sat)] = pd.DataFrame([user_x[sat], user_y[sat], user_z[sat]],
#                                                                 index=['user_x{}'.format(sat), 'user_y{}'.format(sat),
#                                                                        'user_z{}'.format(sat)])
#         globals()['user_vector{}'.format(sat)] = pd.DataFrame([user_location_phi[sat], user_location_theta[sat]],index=['user_location_phi{}'.format(sat), 'user_location_theta{}'.format(sat)])
#
#     with pd.ExcelWriter('./UserLocation.xlsx') as writer:
#         for sat in range(pa.number_of_satellite):
#             globals()['user_location{}'.format(sat)].to_excel(writer, sheet_name='User_location{}'.format(sat))
#     with pd.ExcelWriter('./UserVector.xlsx') as writer:
#         for sat in range(pa.number_of_satellite):
#             globals()['user_vector{}'.format(sat)].to_excel(writer, sheet_name='GroundNetwork{}'.format(sat))
# # -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------사용자 위치 불러오기---------------------------------
    user_x = np.zeros((pa.number_of_satellite, pa.number_of_user))
    user_y = np.zeros((pa.number_of_satellite, pa.number_of_user))
    user_z = np.zeros((pa.number_of_satellite, pa.number_of_user))
    user_location_phi = np.zeros((pa.number_of_satellite, pa.number_of_user))
    user_location_theta = np.zeros((pa.number_of_satellite, pa.number_of_user))
    for sat in range(pa.number_of_satellite):
        read = pd.read_excel('./UserLocation.xlsx', sheet_name='User_location{}'.format(sat))
        vector = pd.read_excel('.//UserVector.xlsx', sheet_name='GroundNetwork{}'.format(sat))
        for user in range(pa.number_of_user):
            user_x[sat][user] = read[user][0]
            user_y[sat][user] = read[user][1]
            user_z[sat][user] = read[user][2]
            user_location_phi[sat][user] = vector[user][0]
            user_location_theta[sat][user] = vector[user][1]
# -----------------------------------------------------------------------------------------------

    Beam_x, Beam_y, Beam_z = env.MultiBeam_create()
    satellite_x, satellite_y, satellite_z = env.initial_satellite_location()
    score_net, score_sat = [], []
    move_to_sat_x, move_to_sat_y, move_to_sat_z = env.move_to_satellite()
    for cycle in range(agent.number_of_cycle):
        # print("cycle : ", cycle)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        current_state = env.initial_state(state_size, satellite_x, satellite_y, satellite_z)
        user_traffic = env.User_requirment(cycle)  # episode 마다 트래픽 변동
        for sat in range(agent.number_of_agent):
            network = agent.handover(cycle, sat)
            timestep = 0
            score_ = 0
            outage_count = 0
            D_reward = []
            # === [추가] fairness/outage/throughput 누적 변수 ===
            fairness_list = []
            outage_list = []
            throughput_list = []
            # ========================================
            # user_traffic = env.User_requirment(cycle)  # episode 마다 트래픽 변동
            while (1):
                state_input = env.set_MDP_state(current_state, network)
                action = agent.get_action(state_input, network)
                next_state, next_state_input = env.set_MDP_action(current_state, action, move_to_sat_x, move_to_sat_y, move_to_sat_z, network, timestep)
                distance_sat2user = env.calculate_distance_sat2user(user_x, user_y, user_z, move_to_sat_x, move_to_sat_y, move_to_sat_z, network, timestep)
                distance_sat2Beam = env.calculate_distance_sat2Beam(Beam_x, Beam_y, Beam_z, move_to_sat_x, move_to_sat_y, move_to_sat_z, network, timestep)
                distance_Beam2user = env.calculate_distance_Beam2user(Beam_x, Beam_y, Beam_z, user_x, user_y, user_z, network)
                # user_x, user_y, user_z, user_location_phi, user_location_theta = env.move_to_user(user_x, user_y, user_z, user_location_phi, user_location_theta, network)
                angle_BSU = env.calculate_angle(distance_sat2Beam, distance_Beam2user, distance_sat2user)
                elevation_angle = env.calculate_elevation_angle(distance_sat2user)
                LoS = env.calculate_LoS_probability(elevation_angle)
                doppler_freq_ = pa.carrier_freq
                pathloss = env.calculate_pathloss(LoS, doppler_freq_, distance_sat2user, elevation_angle)
                # === [수정] elevation_angle, LoS 인자 추가 (Reviewer 1.3/2.1) ===
                channel_gain = env.calculate_channel_gain(angle_BSU, pathloss, doppler_freq_)
                h = env.smallscalefading(elevation_angle, LoS)
                RSSI = env.calculate_RSSI(channel_gain,h, next_state, network)
                SINR = env.calculate_SINR(RSSI, next_state, network)
                sinr = ut.watt2dB(SINR)
                # user_traffic = env.User_requirment() # iteration 마다 트래픽 변동
                ChannelAllocation = env.channel_allocation(SINR, next_state,user_traffic[network], network)
                # === [수정] method='proposed' 인자 추가 (Reviewer 1.2) ===
                ADR,reward, Normalization_reward = env.calculate_reward(ChannelAllocation, SINR, next_state, network, method='proposed')
                # ==========================================================
                agent.append_sample(state_input, action, Normalization_reward, next_state_input, network)
                current_state = next_state
                score_ += reward
                D_reward.append(reward)


                user_rates = np.sum(ADR, axis=0)
                active_user = np.sum(user_traffic[network])
                if np.sum(user_rates) > 0:
                    fairness = (np.sum(user_rates) ** 2) / (active_user * np.sum(user_rates ** 2))
                else:
                    fairness = 0.0
                fairness_list.append(fairness)

                ADR_MIN = 0.1e6  # 1 Mbps
                outage_count = np.sum((user_traffic[network] == 1) & (user_rates < ADR_MIN))
                outage_rate = outage_count / active_user
                outage_list.append(outage_rate)

                # Throughput: 전체 유저 합산 데이터율
                throughput = np.sum(user_rates)
                throughput_list.append(throughput)
                # === [추가] 마지막 episode 유저별 ADR 저장 ===
                if cycle == agent.number_of_cycle - 1:
                    if not hasattr(globals().get('user_adr_net{}'.format(network), None), 'append'):
                        globals()['user_adr_net{}'.format(network)] = []
                    globals()['user_adr_net{}'.format(network)].append(user_rates.copy())
                # ============================================
                # =================================================================

                if len(globals()['memory{}'.format(network)]) > agent.train_start:
                    agent.train_model(network)
                    if timestep % agent.target_update_period == 0 and timestep !=0:
                        agent.update_target_model(network)
                timestep += 1
                if timestep == pa.number_of_iteration:
                    globals()['score_net{}'.format(network)].append(score_/pa.number_of_iteration)
                    globals()['score_sat{}'.format(sat)].append(score_/(pa.number_of_iteration * agent.number_of_agent))
                    # === [추가] episode 평균 fairness/outage 저장 ===
                    if not hasattr(globals().get('fairness_net{}'.format(network), None), 'append'):
                        globals()['fairness_net{}'.format(network)] = []
                        globals()['outage_net{}'.format(network)] = []
                        globals()['throughput_net{}'.format(network)] = []
                    globals()['fairness_net{}'.format(network)].append(np.mean(fairness_list))
                    globals()['outage_net{}'.format(network)].append(np.mean(outage_list))
                    globals()['throughput_net{}'.format(network)].append(np.mean(throughput_list))
                    # =================================================
                    break
            if  cycle % agent.number_of_cycle == 0:
                print("cycle:",cycle,"sat:", sat, "net:", network, "score_net:", globals()['score_net{}'.format(network)][-1])

        # === 중간 저장 (number_of_cycle // 2) ===
        _mid = agent.number_of_cycle // 2 - 1
        if cycle == _mid:
            _n = _mid + 1
            _col = [f'episode{i}' for i in range(_n)]
            _idx = [f'Network{i}' for i in range(agent.number_of_agent)]
            _sn = np.array([globals()[f'score_net{i}'] for i in range(agent.number_of_agent)]).reshape((agent.number_of_agent, _n))
            _ss = np.sum(np.array([globals()[f'score_sat{i}'] for i in range(agent.number_of_agent)]).reshape((agent.number_of_agent, _n)), axis=0).reshape((1, _n))
            _fn = np.array([globals()[f'fairness_net{i}'] for i in range(agent.number_of_agent)]).reshape((agent.number_of_agent, _n))
            _on = np.array([globals()[f'outage_net{i}'] for i in range(agent.number_of_agent)]).reshape((agent.number_of_agent, _n))
            _tn = np.array([globals()[f'throughput_net{i}'] for i in range(agent.number_of_agent)]).reshape((agent.number_of_agent, _n))
            _ts = datetime.now().strftime("%y%m%d")
            with pd.ExcelWriter(f'./Proposed_test{_ts}_mid{_n}.xlsx') as _w:
                pd.DataFrame(_sn, index=_idx, columns=_col).to_excel(_w, sheet_name='score_net')
                pd.DataFrame(_ss, columns=_col).to_excel(_w, sheet_name='score_sat')
                pd.DataFrame(_fn, index=_idx, columns=_col).to_excel(_w, sheet_name='fairness_net')
                pd.DataFrame(_on, index=_idx, columns=_col).to_excel(_w, sheet_name='outage_net')
                pd.DataFrame(_tn, index=_idx, columns=_col).to_excel(_w, sheet_name='throughput_net')
            print(f"=== 중간 저장 완료: cycle {cycle} ({_n} cycles) ===")
        # ==========================================
        for sat in range(agent.number_of_agent):
            globals()['target_model{}'.format(sat)].save_weights('./save_weights/Proposed_test{}{}'.format(sat, _weight_ext))


    for i in range(agent.number_of_agent):
        score_net.append(globals()['score_net{}'.format(i)])
        score_sat.append(globals()['score_sat{}'.format(i)])
    score_net = np.array(score_net)
    score_net = score_net.reshape((agent.number_of_agent, agent.number_of_cycle))
    score_sat = np.array(score_sat)
    score_sat = np.sum(score_sat.reshape((agent.number_of_agent, agent.number_of_cycle)), axis=0).reshape((1, agent.number_of_cycle))
    colum_list=[f'Network{i}' for i in range(0,agent.number_of_agent)]
    colum_list_sat=[f'sat{i}' for i in range(0,agent.number_of_agent)]
    index_list=[f'episode{i}' for i in range(0,agent.number_of_cycle)]

    Score_net = pd.DataFrame(score_net[:], index= colum_list, columns = index_list)
    Score_sat = pd.DataFrame(score_sat[:], columns = index_list)

    # === [추가] Fairness & Outage 결과 저장 ===
    fairness_net = np.array([globals()['fairness_net{}'.format(i)] for i in range(agent.number_of_agent)])
    outage_net = np.array([globals()['outage_net{}'.format(i)] for i in range(agent.number_of_agent)])
    throughput_net = np.array([globals()['throughput_net{}'.format(i)] for i in range(agent.number_of_agent)])
    fairness_net = fairness_net.reshape((agent.number_of_agent, agent.number_of_cycle))
    outage_net = outage_net.reshape((agent.number_of_agent, agent.number_of_cycle))
    throughput_net = throughput_net.reshape((agent.number_of_agent, agent.number_of_cycle))
    Fairness_net = pd.DataFrame(fairness_net[:], index=colum_list, columns=index_list)
    Outage_net = pd.DataFrame(outage_net[:], index=colum_list, columns=index_list)
    Throughput_net = pd.DataFrame(throughput_net[:], index=colum_list, columns=index_list)
    # ==========================================

    timestamp = datetime.now().strftime("%y%m%d")
    with pd.ExcelWriter('./Proposed_test{}.xlsx'.format(timestamp)) as writer:
        Score_net.to_excel(writer, sheet_name='score_net')
        Score_sat.to_excel(writer, sheet_name='score_sat')
        # === [추가] fairness/outage 시트 추가 ===
        Fairness_net.to_excel(writer, sheet_name='fairness_net')
        Outage_net.to_excel(writer, sheet_name='outage_net')
        Throughput_net.to_excel(writer, sheet_name='throughput_net')
        # === [추가] 유저별 ADR 저장 ===
        for i in range(agent.number_of_agent):
            user_adr = np.array(globals()['user_adr_net{}'.format(i)])
            User_ADR = pd.DataFrame(user_adr,
                                    index=[f'iter{t}' for t in range(pa.number_of_iteration)],
                                    columns=[f'user{u}' for u in range(pa.number_of_user)])
            User_ADR.to_excel(writer, sheet_name=f'user_adr_net{i}')
        # ==============================
        # ========================================
