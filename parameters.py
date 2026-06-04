import numpy as np

# ----------------- dynamic parameters -----------------
number_of_user = 64
number_of_beam = 8
kind_of_frequency_reuse = [1, 2, 4, 8]
move_radius = 3 # [m]
number_of_iteration = 1000

# ----------------- 환경 파라미터 ---------------
number_of_satellite = 49
number_of_area = 49
carrier_freq = 2*10**9  # Hz
bandwidth = 20 * 10 ** 6  # Hz
number_of_channel = 8
radius = 6400*10**3  # [m] 지구 반지름
altitude = 1200*10**3  # [m] 위성 고도
velocity_of_sat = 7240  # m/sec
velocity_of_light = 3*10**8  # m/sec

# ----------------- 시뮬레이션 파라미터 ----------------

total_step = number_of_iteration * number_of_satellite
Round = number_of_iteration * number_of_area
Nphi = np.linspace(0, np.pi, number_of_iteration * number_of_area)
Ntheta = np.linspace(0, 2 * np.pi, number_of_iteration * number_of_area)
cycle_of_revolution = 91 * 60  # [sec]

satellite_max_location = radius + altitude
max_transmission_power = 200 / number_of_beam
kind_of_power = [0, (max_transmission_power / 4), (max_transmission_power / 2),
                      (max_transmission_power * 3 / 4), (max_transmission_power)]

antenna_efficiency = 0.55
antenna_radius = 1  # m
antenna_area = np.pi * antenna_radius ** 2
angle_3dB = np.deg2rad(4.4127)
user_RXgain_max = 0  # dBi

# =====================================================================
# === [추가] ISL transfer cost parameters (Reviewer 1.2) ==============
# =====================================================================
P_ISL_TX = 10.0           # W, ISL 송신 전력 (MAAC-IILP 인용)
R_ISL = 10e6              # bps, ISL 채널 용량
TAU_STEP = 0.1            # s, 1 timestep 물리 시간 가정
# Transfer 데이터 양 (bits)
# weight: 12*64+64 + 64*64+64 + 64*19+19 = 6227 params × 32 bit
# buffer: 20000 transitions × (12+1+1+12) floats × 32 bit
WEIGHT_BITS = 6227 * 32
BUFFER_BITS = 10000 * (12 + 1 + 1 + 12) * 32
BITS_PER_STEP = R_ISL * TAU_STEP  # bits per step = 1 Mbit

net_type = [0, 1, 2, 0, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 0, 2, 1, 0, 1, 2, 0, 2, 2, 0, 2, 2, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 2, 2, 1, 0, 2, 2, 0, 2, 2, 0, 1, 2]
active_ratio = {0: 0.9, 1: 0.1, 2: 0.6}
urban_sigma_phi = 0.016
urban_sigma_theta = 0.032


P_ISL = {
    'proposed':      P_ISL_TX * (WEIGHT_BITS + BUFFER_BITS) / (BITS_PER_STEP * number_of_iteration),
    'weights_based': P_ISL_TX * WEIGHT_BITS / (BITS_PER_STEP * number_of_iteration),
    'dataset_based': P_ISL_TX * BUFFER_BITS / (BITS_PER_STEP * number_of_iteration),
    'no_collab':     0.0,
    'sota':    P_ISL_TX * 2*WEIGHT_BITS / (BITS_PER_STEP * number_of_iteration),
    'always_on':     0.0,
    'random':        0.0,
}

# =====================================================================
# === [추가] Rician fading + Outdated CSI parameters (Reviewer 1.3/2.1)
# =====================================================================
RHO_D = 0 


K_FACTOR_TABLE = {
    10: (11.40, 6.26),
    20: (19.45, 10.32),
    30: (20.80, 16.34),
    40: (21.60, 15.63),
    50: (21.60, 14.22),
    60: (19.75, 14.19),
    70: (12.00, 5.70),
    80: (12.85, 9.91),
    90: (12.85, 9.91),
}
