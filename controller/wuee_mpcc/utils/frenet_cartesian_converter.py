import numpy as np

from utils.splinify import SplineTrack


def convert_frenet_to_cartesian(spline, frenet_coords) -> np.array:

    # ### HJ : 원본은 R = rotate(-π/2) → +d 가 path 오른쪽. 하지만 bicycle_model.py
    # 의 dynamics dn_dot = vx·sin(α)+vy·cos(α) 는 +n = LEFT convention (body +y=LEFT
    # 일 때 dn_dot>0). 결과: MPCC state 의 n 값을 rendering 할 때 좌우 반전 →
    # prediction 마커가 path 반대편에 평행 이동되어 그려지고, boundary 마커도 틀린 측에.
    # +π/2 로 회전해 LEFT 방향으로 맞춘다 (race_stack /car_state/odom_frenet 도 +n=LEFT).
    co, si = np.cos(np.pi / 2), np.sin(np.pi / 2)
    R = np.array(((co, -si), (si, co)))

    if frenet_coords.shape[0] > 2:
        return np.array(
            [
                spline.get_coordinate(s) + d * spline.get_derivative(s).reshape(1, 2) @ R
                for s, d in frenet_coords
            ]
        ).reshape(-1, 2)

    else:

        s = frenet_coords[0]
        d = frenet_coords[1]

        return spline.get_coordinate(s) + d * spline.get_derivative(s) @ R
