from enum import IntEnum


class State(IntEnum):
    POS_ON_CENTER_LINE_S = 0
    MIN_DIST_TO_CENTER_LINE_N = 1
    ORIENTATION_ALPHA = 2
    VELOCITY_VX = 3
    VELOCITY_VY = 4
    YAW_RATE_OMEGA = 5
    DUTY_CYCLE_D = 6
    STEERING_ANGLE_DELTA = 7
    PROGRESS_THETA = 8


class Input(IntEnum):
    D_DUTY_CYCLE = 0
    D_STEERING_ANGLE = 1
    D_PROGRESS = 2


class Parameter(IntEnum):
    m = 0
    C1 = 1
    C2 = 2
    CSf = 3
    CSr = 4
    Cr0 = 5
    Cr2 = 6
    Cr3 = 7
    Iz = 8
    lr = 9
    lf = 10
    Bf = 11
    Cf = 12
    Df = 13
    Br = 14
    Cr = 15
    Dr = 16
    Imax_c = 17
    Caccel = 18
    Cdecel = 19
    qc = 20
    ql = 21
    gamma = 22    # 원본 typo (=2) 수정. 실제로는 22 가 맞음.
    r1 = 23
    r2 = 24
    r3 = 25
    # Damping weights — 실차 안정성 위해 추가 (HJ 2026-04-22).
    q_omega = 26  # yaw-rate 페널티
    q_delta = 27  # steer-centering
    q_vy = 28     # 측면 슬립 억제
    # race_stack 정합 Pacejka 계수 추가.
    Ef = 29
    Er = 30
    mu_tire = 31
