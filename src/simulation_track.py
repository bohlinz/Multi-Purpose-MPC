import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
import json


if __name__ == '__main__':


    map_json = "maps/reference_path.json"              
    with open(map_json, 'r') as f:
        map_data = json.load(f)

    wp_x = map_data["x"]
    wp_y = map_data["y"]
    nl = map_data["e_left"]
    nr = map_data["e_right"]
    s = map_data["s"]
    psi = map_data["course"]
    kappa = map_data["rho"]


    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # 直接构建Reference
    reference_path = ReferencePath(wp_x, wp_y, path_resolution,
                                   smoothing_distance=5, max_width=10.0,
                                   circular=True, left_dis=nl, right_dis=nr, s=s, psi=psi, kappa=kappa)
    # Instantiate motion model
    car = BicycleModel(length=3.0, width=1.8,
                       reference_path=reference_path, Ts=0.05)

    
    ##############
    # Controller #
    ##############

    N = 30
    Q = sparse.diags([1.0, 0.0, 0.0])
    R = sparse.diags([0.5, 0.0])
    QN = sparse.diags([1.0, 0.0, 0.0])

    v_max = 100.0  # m/s
    delta_max = 0.66  # rad
    ay_max = 20.0  # m/s^2
    InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                        'umax': np.array([v_max, np.tan(delta_max)/car.length])}
    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

    # Compute speed profile
    a_min = -30.0  # m/s^2
    a_max = 10  # m/s^2
    SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                               'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
    car.reference_path.compute_speed_profile(SpeedProfileConstraints)

    ##############
    # Simulation #
    ##############

    # Set simulation time to zero
    t = 0.0

    # Logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]
    v_log = [0.0]

    # Until arrival at end of path
    while car.s < reference_path.length:

        # Get control signals
        u = mpc.get_control()

        # Simulate car
        car.drive(u)

        # Log car state
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)
        v_log.append(u[0])
        print(car.temporal_state)

        # Increment simulation time
        t += car.Ts

        # Plot path and drivable area
        # reference_path.show()

        # Plot car
        car.show()

        # Plot MPC prediction
        mpc.show_prediction()

        # Set figure title
        plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.format(u[0], u[1], t))
        plt.axis('off')
        plt.pause(0.001)
