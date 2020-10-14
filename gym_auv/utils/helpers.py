import numpy as np
import gym_auv.utils.geomutils as geom

def generate_obstacle(rng, path, vessel, displacement_dist_std=150, obst_radius_distr=np.random.poisson, obst_radius_mean=30):
    min_distance = 0
    while min_distance <= 0:
        obst_displacement_dist = np.random.normal(0, displacement_dist_std)
        obst_arclength = (0.1 + 0.8*rng.rand())*path.length
        obst_position = path(obst_arclength)
        obst_displacement_angle = geom.princip(path.get_direction(obst_arclength) - np.pi/2)
        obst_position += obst_displacement_dist*np.array([
            np.cos(obst_displacement_angle), 
            np.sin(obst_displacement_angle)
        ])
        obst_radius = max(1, obst_radius_distr(obst_radius_mean))

        vessel_distance_vec = geom.Rzyx(0, 0, -vessel.heading).dot(
            np.hstack([obst_position - vessel.position, 0])
        )
        vessel_distance = np.linalg.norm(vessel_distance_vec) - vessel.width - obst_radius
        goal_distance = np.linalg.norm(obst_position - path(path.length)) - obst_radius
        min_distance = min(vessel_distance, goal_distance)

    return (obst_position, obst_radius)