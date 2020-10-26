import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import minimize
import shapely.geometry
from gym_auv.objects.obstacles import CircularObstacle, PolygonObstacle, VesselObstacle
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.envs import RealWorldEnv

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 0.5
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib import animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import gym_auv

PLOT_COLOR_TRESHOLD = 500
SHADOW_LENGTH = 100

#matplotlib.use('pdf')

def report(env, report_dir, lastn=100):
    try:
        os.makedirs(report_dir, exist_ok=True)

        if lastn > -1:
            relevant_history = env.history[-min(lastn, len(env.history)):]
        else:
            relevant_history = env.history

        collisions = np.array([obj['collision'] for obj in relevant_history])
        no_collisions = collisions == 0
        cross_track_errors = np.array([obj['cross_track_error'] for obj in relevant_history])
        progresses = np.array([obj['progress'] for obj in relevant_history])
        rewards = np.array([obj['reward'] for obj in relevant_history])
        timesteps = np.array([obj['timesteps'] for obj in relevant_history])
        duration = np.array([obj['duration'] for obj in relevant_history])
        pathlengths = np.array([obj['pathlength'] for obj in relevant_history])
        speeds = np.array([obj['pathlength']/obj['duration'] if obj['duration'] > 0 else np.nan for obj in relevant_history])

        with open(os.path.join(report_dir, 'report.txt'), 'w') as f:
            f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(lastn))
            f.write('{:<30}{:<30}\n'.format('Episodes', len(pathlengths)))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Reward', rewards.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Std. Reward', rewards.std()))
            f.write('{:<30}{:<30.2%}\n'.format('Avg. Progress', progresses.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Collisions', collisions.mean()))
            f.write('{:<30}{:<30.2%}\n'.format('No Collisions', no_collisions.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Cross-Track Error', cross_track_errors.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Timesteps', timesteps.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Duration', duration.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Pathlength', pathlengths.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Speed', speeds.mean()))

        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        #plt.rc('font', family='serif', serif='Times')
        #plt.rc('text', usetex=True) #RAISES FILENOTFOUNDERROR
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)

        collisions = np.array([obj['collision'] for obj in env.history])
        smoothed_collisions = gaussian_filter1d(collisions.astype(float), sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(collisions, color='blue', linewidth=0.5, alpha=0.2, label='Collisions')
        ax.plot(smoothed_collisions, color='blue', linewidth=1, alpha=0.4)
        ax.set_title('Collisions')
        ax.set_ylabel(r"Collisions")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'collisions.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

        cross_track_errors = np.array([obj['cross_track_error'] for obj in env.history])
        smoothed_cross_track_errors = gaussian_filter1d(cross_track_errors, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(cross_track_errors, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_cross_track_errors, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Avg. Cross-Track Error")
        ax.set_xlabel(r"Episode")
        #ax.legend()
        fig.savefig(os.path.join(report_dir, 'cross_track_error.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

        rewards = np.array([obj['reward'] for obj in env.history])
        smoothed_rewards = gaussian_filter1d(rewards, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rewards, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_rewards, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Reward")
        ax.set_xlabel(r"Episode")
        #ax.legend()
        fig.savefig(os.path.join(report_dir, 'reward.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

        progresses = np.array([obj['progress'] for obj in env.history])
        smoothed_progresses = gaussian_filter1d(progresses, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax.plot(progresses, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_progresses, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Progress [%]")
        ax.set_xlabel(r"Episode")
        #ax.legend()
        fig.savefig(os.path.join(report_dir, 'progress.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

        timesteps = np.array([obj['timesteps'] for obj in env.history])
        smoothed_timesteps = gaussian_filter1d(timesteps.astype(float), sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(timesteps, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_timesteps, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Timesteps")
        ax.set_xlabel(r"Episode")
        #ax.legend()
        fig.savefig(os.path.join(report_dir, 'timesteps.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

        plt.clf()

    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))


def plot_trajectory(env, fig_dir, local=False, size=100, fig_prefix='', episode_dict=None):
    """
    Plots the result of a path following episode.

    Parameters
    ----------
    fig_dir : str
        Absolute path to a directory to store the plotted
        figure in.
    """

    #print('Plotting with local = ' + str(local))

    path = env.last_episode['path']
    path_taken = env.last_episode['path_taken']
    obstacles = env.last_episode['obstacles']

    #np.savetxt(os.path.join(fig_dir, 'path_taken.txt'), path_taken) 

    if (fig_prefix != '' and not fig_prefix[0] == '_'):
        fig_prefix = '_' + fig_prefix

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1.0)

    if local:
        axis_min_x = env.vessel.position[0] - size
        axis_max_x = env.vessel.position[0] + size
        axis_min_y = env.vessel.position[1] - size
        axis_max_y = env.vessel.position[1] + size 

    else:
        axis_min_x = path[0, :].min() - 200
        axis_max_x = path[0, :].max() + 200
        axis_min_y = path[1, :].min() - 200
        axis_max_y = path[1, :].max() + 200
        daxisx = axis_max_x - axis_min_x
        daxisy = axis_max_y - axis_min_y
        if daxisx > daxisy:
            d = daxisx - daxisy
            axis_min_y -= d/2
            axis_max_y += d/2
        if daxisx < daxisy:
            d = daxisy - daxisx
            axis_min_x -= d/2
            axis_max_x += d/2
        axis_max = max(axis_max_x, axis_max_y)
        axis_min = min(axis_min_x, axis_min_y)

    for obst in obstacles:
        if isinstance(obst, CircularObstacle):
            obst_object = plt.Circle(
                obst.position,
                obst.radius,
                facecolor='tab:red',
                edgecolor='black',
                linewidth=0.5,
                zorder=10
            )
            obst_object.set_hatch('////')
            obst = ax.add_patch(obst_object)
        elif isinstance(obst, PolygonObstacle):
            obst_object = plt.Polygon(
                np.array(obst.points), True,
                facecolor='#C0C0C0',
                edgecolor='black',
                linewidth=0.5,
                zorder=10
            )
            obst = ax.add_patch(obst_object)

    for obst in obstacles:
        if not obst.static:
            # if isinstance(obst, VesselObstacle):
            #     x_arr = [elm[0] for elm in obst.path_taken[len(obst.path_taken) - min(len(obst.path_taken), 100):]]
            #     y_arr = [elm[1] for elm in obst.path_taken[len(obst.path_taken) - min(len(obst.path_taken), 100):]]

            #     points = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
            #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #     colors = []
            #     for i in range(len(x_arr)-1):
            #         di = len(x_arr)-1 - i
            #         if di > PLOT_COLOR_TRESHOLD:
            #             c = (1.0, 0.0, 0.0)
            #         else:
            #             c = (min(1.0, di*0.01), max(0.0, 1.0-di*0.01), 0.0)
            #         colors.append(c)
            #     lc = LineCollection(segments, color=colors, linewidth=1.0, linestyle='--')
            #     ax.add_collection(lc)

            #     #ax.plot(x_arr, y_arr, dashes=[6, 2], color='red', linewidth=0.5, alpha=0.3)

            if (not local) and isinstance(obst, VesselObstacle):
                x_arr = [elm[1][0] for elm in obst.trajectory]
                y_arr = [elm[1][1] for elm in obst.trajectory]
                ax.plot(x_arr, y_arr, dashes=[6, 2], color='red', linewidth=0.5, alpha=0.3)

            plt.arrow(
                obst.init_boundary.centroid.coords[0][0],
                obst.init_boundary.centroid.coords[0][1],
                120*obst.dx,
                120*obst.dy,
                head_width=3 if local else 8,
                color='black',
                zorder=9
            )

    for obst in obstacles:
        if isinstance(obst, VesselObstacle):
            if local and (abs(obst.position[0] - env.vessel.position[0]) > size or abs(obst.position[1] - env.vessel.position[1]) > size):
                continue
            if local:
                vessel_obst = VesselObstacle(
                    width=obst.width, 
                    trajectory=[],  
                    init_position=obst.position,
                    init_heading=obst.heading, 
                    init_update=False
                )
                vessel_obst_object = plt.Polygon(
                    np.array(list(vessel_obst.boundary.exterior.coords)), True,
                    facecolor='#C0C0C0',
                    edgecolor='red',
                    linewidth=0.5,
                    zorder=10
                )
            else:
                vessel_obst = VesselObstacle(
                    width=obst.width, 
                    trajectory=[],  
                    init_position=obst.init_position,
                    init_heading=obst.heading, 
                    init_update=False
                )
                vessel_obst_object = plt.Polygon(
                    np.array(list(vessel_obst.init_boundary.exterior.coords)), True,
                    facecolor='#C0C0C0',
                    edgecolor='red',
                    linewidth=0.5,
                    zorder=10
                )
            ax.add_patch(vessel_obst_object)
            if local and len(obst.heading_taken) >= SHADOW_LENGTH:
                position = obst.path_taken[-SHADOW_LENGTH]
                heading = obst.heading_taken[-SHADOW_LENGTH]

                vessel_obst = VesselObstacle(
                    width=obst.width, 
                    trajectory=[],  
                    init_position=position,
                    init_heading=heading, 
                    init_update=False
                )
                vessel_obst_object = plt.Polygon(
                    np.array(list(vessel_obst.boundary.exterior.coords)), True,
                    facecolor='none',
                    edgecolor='red',
                    linewidth=0.5,
                    linestyle='--',
                    zorder=10
                )
                ax.add_patch(vessel_obst_object)

                # x_arr = [elm[0] for elm in obst.path_taken[len(obst.path_taken) - min(len(obst.path_taken), SHADOW_LENGTH):]]
                # y_arr = [elm[1] for elm in obst.path_taken[len(obst.path_taken) - min(len(obst.path_taken), SHADOW_LENGTH):]]
                # points = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
                # segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # colors = []
                # for i in range(len(x_arr)-1):
                #     di = len(x_arr)-1 - i
                #     if di > PLOT_COLOR_TRESHOLD:
                #         c = (1.0, 0.0, 0.0)
                #     else:
                #         c = (max(0.0, 1-di*0.01), 0.0, 0.0)
                #     colors.append(c)
                # lc = LineCollection(segments, color=colors, linewidth=0.5, linestyle='--', zorder=9)
                # ax.add_collection(lc)

    if local:
        vessel_obst = VesselObstacle(
            width=env.vessel.width, 
            trajectory=[],  
            init_position=env.vessel.position,
            init_heading=env.vessel.heading, 
            init_update=False
        )
        vessel_obst_object = plt.Polygon(
            np.array(list(vessel_obst.boundary.exterior.coords)), True,
            facecolor='#C0C0C0',
            edgecolor='red',
            linewidth=0.5,
            zorder=10
        )
        ax.add_patch(vessel_obst_object)

        if len(env.vessel.heading_taken) >= SHADOW_LENGTH:
            position = env.vessel.path_taken[-SHADOW_LENGTH]
            heading = env.vessel.heading_taken[-SHADOW_LENGTH]

            vessel_obst = VesselObstacle(
                width=env.vessel.width, 
                trajectory=[],  
                init_position=position,
                init_heading=heading, 
                init_update=False
            )
            vessel_obst_object = plt.Polygon(
                np.array(list(vessel_obst.boundary.exterior.coords)), True,
                facecolor='none',
                edgecolor='red',
                linewidth=0.5,
                linestyle='--',
                zorder=10
            )
            ax.add_patch(vessel_obst_object)

    if local and size <= 50:
        ax.set_ylabel(r"North (m)")
        ax.set_xlabel(r"East (m)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
    else:
        ax.set_ylabel(r"North (km)")
        ax.set_xlabel(r"East (km)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y/100)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y/100)))
    ax.set_xlim(axis_min_x, axis_max_x)
    ax.set_ylim(axis_min_y, axis_max_y)
    #ax.legend()

    dashmultiplier = 3 if local == False else size/100
    linemultiplier = 1 if local == False else size/300
    ax.plot(path[0, :], path[1, :], dashes=[3*dashmultiplier, 1*dashmultiplier], color='black', linewidth=1.0*linemultiplier, label=r'Path', zorder=8)

    if episode_dict is None or local:
        pathcolor = 'red'
        L = len(env.vessel.heading_taken)
        if L + 5 >= SHADOW_LENGTH:
            ax.plot(path_taken[:L-SHADOW_LENGTH-7, 0], path_taken[:L-SHADOW_LENGTH-7, 1], dashes=[1*dashmultiplier, 2*dashmultiplier], color=pathcolor, linewidth=1.0*linemultiplier, label=r'Path taken')
            ax.plot(path_taken[L-SHADOW_LENGTH+7+int(env.vessel.width):, 0], path_taken[L-SHADOW_LENGTH+7+int(env.vessel.width):, 1],  dashes=[1*dashmultiplier, 2*dashmultiplier], color=pathcolor, linewidth=1.0*linemultiplier, label=r'Path taken')
        
        else:
            ax.plot(path_taken[:, 0], path_taken[:, 1], dashes=[1*dashmultiplier, 2*dashmultiplier], color=pathcolor, linewidth=1.0*linemultiplier, label=r'Path taken')

        # x_arr = path_taken[:, 0]
        # y_arr = path_taken[:, 1]
        # points = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # colors = []
        # for i in range(len(x_arr)-1):
        #     di = len(x_arr)-1 - i
        #     if di > PLOT_COLOR_TRESHOLD:
        #         c = (1.0, 0.0, 0.0, 0.3)
        #     else:
        #         c = (max(0.0, 1-di/PLOT_COLOR_TRESHOLD), 0.0, 0.0, min(1.0, 0.3+di*0.7/PLOT_COLOR_TRESHOLD))
        #     colors.append(c)
        # lc = LineCollection(segments, color=colors, linewidth=1.0, linestyle='--', zorder=9)
        # ax.add_collection(lc)
        
    else:
        episode_dict_colors = [episode_dict[value_key][1] for value_key in episode_dict]
        normalize = mcolors.Normalize(vmin=min(episode_dict_colors), vmax=max(episode_dict_colors))
        colormap = cm.coolwarm
        for value_key in episode_dict:
            value_path_taken = episode_dict[value_key][0]['path_taken']
            ax.plot(value_path_taken[:, 0], value_path_taken[:, 1], color=colormap(normalize(episode_dict[value_key][1])))

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(episode_dict_colors)
        cbar = fig.colorbar(scalarmappaple)
        cbar.set_label(r'$ \log10\;{\lambda}$')

    if isinstance(env, RealWorldEnv) and not local:
        for x, y in zip(*env.path.init_waypoints):
            waypoint_marker = plt.Circle(
                (x, y),
                (axis_max - axis_min)/150,
                facecolor='red',
                linewidth=0.5,
                zorder=11
            )
            ax.add_patch(waypoint_marker)

    if not local:
        ax.annotate("Goal", 
            xy=(path[0, -1], path[1, -1] + (axis_max - axis_min)/25),   
            fontsize=11, ha="center", zorder=20, color='white', family='sans-serif',
            bbox=dict(facecolor='tab:red', edgecolor='black', alpha=0.75, boxstyle='round')
        )
        ax.annotate("Start", 
            xy=(path[0, 0], path[1, 0] - (axis_max - axis_min)/20),
            fontsize=11, ha="center", zorder=20, color='white', family='sans-serif',
            bbox=dict(facecolor='tab:red', edgecolor='black', alpha=0.75, boxstyle='round')
        )

    fig.savefig(os.path.join(fig_dir, '{}path.pdf'.format(fig_prefix)), format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_scenario(env, fig_dir, fig_postfix='', show=True):
    path = env.path(np.linspace(0, env.path.length, 1000))

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1.0)

    axis_min_x = path[0, :].min() - 200
    axis_max_x = path[0, :].max() + 200
    axis_min_y = path[1, :].min() - 200
    axis_max_y = path[1, :].max() + 200
    daxisx = axis_max_x - axis_min_x
    daxisy = axis_max_y - axis_min_y
    if daxisx > daxisy:
        d = daxisx - daxisy
        axis_min_y -= d/2
        axis_max_y += d/2
    if daxisx < daxisy:
        d = daxisy - daxisx
        axis_min_x -= d/2
        axis_max_x += d/2
    axis_max = max(axis_max_x, axis_max_y)
    axis_min = min(axis_min_x, axis_min_y)

    for obst in env.obstacles:
        #if isinstance(obst, CircularObstacle):
        if not obst.static:
            
            if isinstance(obst, VesselObstacle):
                x_arr = [elm[1][0] for elm in obst.trajectory]
                y_arr = [elm[1][1] for elm in obst.trajectory]
                ax.plot(x_arr, y_arr, dashes=[6, 2], color='red', linewidth=0.5, alpha=0.3)

            plt.arrow(
                obst.boundary.centroid.coords[0][0],
                obst.boundary.centroid.coords[0][1],
                120*obst.dx,
                120*obst.dy,
                head_width=8,
                color='black',
                zorder=11
            )

    for obst in env.obstacles:
        if isinstance(obst, CircularObstacle):
            obst_object = plt.Circle(
                obst.position,
                obst.radius,
                facecolor='tab:red',
                edgecolor='black',
                linewidth=0.5,
                zorder=10
            )
            obst_object.set_hatch('////')
        elif isinstance(obst, PolygonObstacle):
            obst_object = plt.Polygon(
                np.array(obst.points), True,
                facecolor='#C0C0C0',
                edgecolor='black',
                linewidth=0.5,
            )
        elif isinstance(obst, VesselObstacle):
            obst_object = plt.Polygon(
                np.array(list(obst.boundary.exterior.coords)), True,
                facecolor='#C0C0C0',
                edgecolor='red',
                linewidth=0.5,
            )
        obst = ax.add_patch(obst_object)

    
    ax.set_ylabel(r"North (km)")
    ax.set_xlabel(r"East (km)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y/100)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y/100)))
    ax.set_xlim(axis_min_x, axis_max_x)
    ax.set_ylim(axis_min_y, axis_max_y)
    #ax.legend()

    ax.plot(path[0, :], path[1, :], dashes=[6, 2], color='black', linewidth=1.5)
    # if isinstance(env, RealWorldEnv):
    #     for x, y in zip(*env.path.init_waypoints):
    #         waypoint_marker = plt.Circle(
    #             (x, y),
    #             (axis_max - axis_min)/150,
    #             facecolor='red',
    #             linewidth=0.5,
    #             zorder=11,
    #         )
    #         ax.add_patch(waypoint_marker)
    ax.annotate("Goal", 
        xy=(path[0, -1], path[1, -1] + (axis_max - axis_min)/25),
        fontsize=11, ha="center", zorder=20, color='white', family='sans-serif',
        bbox=dict(facecolor='tab:red', edgecolor='black', alpha=0.75, boxstyle='round')
    )
    ax.annotate("Start", 
        xy=(path[0, 0], path[1, 0] - (axis_max - axis_min)/20),
        fontsize=11, ha="center", zorder=20, color='white', family='sans-serif',
        bbox=dict(facecolor='tab:red', edgecolor='black', alpha=0.75, boxstyle='round')
    )

    fig.savefig(os.path.join(fig_dir, 'Scenario_{}.pdf'.format(fig_postfix)), format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def plot_actions(env, agent, fig_dir, fig_prefix='', N=500, creategifs=True, createpdfs=True):
    
    env.vessel.reset([0, 0, 0])
    OBST_RADIUS = 10
    theta_arr = np.linspace(-np.pi/2, np.pi/2, N)
    r_arr = np.linspace(1, 100, N)
    Theta, R = np.meshgrid(theta_arr, r_arr)
    surge_arr = np.zeros(Theta.shape)
    steer_arr = np.zeros(Theta.shape)

    for i_t, theta in enumerate(theta_arr):
        for i_r, r in enumerate(r_arr):
            position = (np.cos(theta)*(r + OBST_RADIUS), np.sin(theta)*(r + OBST_RADIUS))
            env.obstacles = [CircularObstacle(position, OBST_RADIUS)]
            obs = env.observe()
            action, _states = agent.predict(obs, deterministic=True)
            action[0] = (action[0] + 1)/2 
            surge = env.vessel._surge(action[0])
            steer = 180/np.pi*env.vessel._steer(action[1])
            steer_arr[i_r, i_t] = steer 
            surge_arr[i_r, i_t] = surge 

            sys.stdout.write('Plotting progress: {:.2%}\r'.format((i_t*N + i_r + 1)/N**2))
            sys.stdout.flush()
    print('\t'*10)

    np.save(os.path.join(fig_dir, 'Theta'), Theta)
    np.save(os.path.join(fig_dir, 'R'), R)
    np.save(os.path.join(fig_dir, 'steer_arr'), steer_arr)
    np.save(os.path.join(fig_dir, 'surge_arr'), steer_arr)

    env.close()

    if creategifs:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        steer_surf = ax.plot_surface(Theta, R, steer_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Angle")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}°'.format(y*180/np.pi)))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}°'.format(y)))
        fig.colorbar(steer_surf, shrink=0.5, aspect=5)
        print('Steering: Creating gif')
        def rotate(angle):
            ax.view_init(azim=angle)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
        rot_animation.save(os.path.join(fig_dir, fig_prefix + 'obst_avoidance_steer.gif'), dpi=80, writer='imagemagick')

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surge_surf = ax.plot_surface(Theta, R, surge_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Angle")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}°'.format(y*180/np.pi)))
        fig.colorbar(surge_surf, shrink=0.5, aspect=5)
        print('Surge: Creating gif')
        def rotate(angle):
            ax.view_init(azim=angle)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
        rot_animation.save(os.path.join(fig_dir, fig_prefix + 'obst_avoidance_surge.gif'), dpi=80, writer='imagemagick')

    if createpdfs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        surge_plot = ax.contourf(Theta, R, surge_arr, levels=20, cmap='coolwarm')
        cbar = fig.colorbar(surge_plot)
        cbar.set_label(r"Propeller thrust force [N]")
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Angle")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}°'.format(y*180/np.pi)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f} m'.format(y)))
        plt.savefig(os.path.join(fig_dir, fig_prefix + 'policy_plot_thrust_contour.pdf'), bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = fig.gca(projection='3d')
        ax.plot_surface(Theta, R, surge_arr, cmap='coolwarm')
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Angle")
        ax.set_zlabel(r"Propeller thrust force")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}m'.format(y)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}°'.format(y*180/np.pi)))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2f}N'.format(y)))
        plt.savefig(os.path.join(fig_dir, fig_prefix + 'policy_plot_thrust_surface.pdf'), bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        steering_plot = ax.contourf(Theta, R, steer_arr, levels=20, cmap='coolwarm')
        cbar = fig.colorbar(steering_plot)
        cbar.set_label(r"Rudder angle [deg]")
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Angle")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}°'.format(y*180/np.pi)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f} m'.format(y)))
        plt.savefig(os.path.join(fig_dir, fig_prefix + 'policy_plot_steering_contour.pdf'), bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = fig.gca(projection='3d')
        ax.plot_surface(Theta, R, steer_arr, cmap='coolwarm')
        ax.set_ylabel(r"Obstacle Distance")
        ax.set_xlabel(r"Relative Obstacle Theta")
        ax.set_zlabel(r"Rudder angle")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}m'.format(y)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}°'.format(y*180/np.pi)))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}°'.format(y*180/np.pi)))
        plt.savefig(os.path.join(fig_dir, fig_prefix + 'policy_plot_steering_surface.pdf'), bbox_inches='tight')

def plot_streamlines(env, agent, fig_dir, fig_prefix='', N=11):
    OBST_POSITION = [0, 50]
    OBST_RADIUS = 25
    env.reset()

    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    waypoints = np.vstack([[0, 0], [0, 100]]).T
    env.path = Path(waypoints)
    env.obstacles = [CircularObstacle(OBST_POSITION, OBST_RADIUS)]
    env.config["min_goal_distance"] = 0
    env.config["min_goal_progress"] = 0
    path = env.path(np.linspace(0, env.path.length, 1000))

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect(1.0)

    axis_min_x = path[0, :].min() - 75
    axis_max_x = path[0, :].max() + 75
    axis_min_y = path[1, :].min() - 25
    axis_max_y = path[1, :].max() + 25
    axis_max = max(axis_max_x, axis_max_y)
    axis_min = min(axis_min_x, axis_min_y)

    ax.plot(path[0, :], path[1, :], color='black', linewidth=1.5, label=r'Path')
    ax.set_ylabel(r"North (m)")
    ax.set_xlabel(r"East (m)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
    ax.set_xlim(axis_min_x, axis_max_x)
    ax.set_ylim(axis_min_y, axis_max_y)

    for i, dx in enumerate(np.linspace(-2*OBST_RADIUS, 2*OBST_RADIUS, N)):
        env.vessel.reset([dx, 0, np.pi/2])

        obs = env.observe()
        while 1:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done or info['progress'] >= 1.0:
                break

            sys.stdout.write('Simulating episode {}/{}, progress {:.2%}\r'.format((i+1), N, info['progress']))
            sys.stdout.flush()

        path_taken = env.vessel.path_taken
        ax.plot(path_taken[:, 0], path_taken[:, 1], dashes=[6, 2], color='red', linewidth=1.0, label=r'$x_0 = ' + str(dx) + r'$')
    
    for obst in env.obstacles:
        circle = plt.Circle(
            obst.position,
            obst.radius,
            facecolor='tab:red',
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )
        obst = ax.add_patch(circle)
        obst.set_hatch('////')

    goal = plt.Circle(
        (path[0, -1], path[1, -1]),
        (axis_max - axis_min)/100,
        facecolor='black',
        linewidth=0.5,
        zorder=11
    )
    ax.add_patch(goal)
    ax.annotate("Goal", 
        xy=(path[0, -1] + (axis_max - axis_min)/15, path[1, -1]), 
        fontsize=12, ha="center", zorder=20, color='black',
    )
    #ax.legend()

    fig.savefig(os.path.join(fig_dir, fig_prefix+'streamlines.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_vector_field(env, agent, fig_dir, fig_prefix='', xstep=2.0, ystep=5.0, obstacle=True):
    OBST_POSITION = [0, 50]
    OBST_RADIUS = 10
    if obstacle:
        obstacles = [CircularObstacle(OBST_POSITION, OBST_RADIUS)]
    else:
        obstacles = []
    env.reset()

    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    waypoints = np.vstack([[0, 0], [0, 100]]).T
    env.config["min_goal_distance"] = 0
    env.config["min_goal_progress"] = 1
    env.config["sensor_rotation"] = False
    env.path = Path(waypoints)
    env.obstacles = obstacles
    path = env.path(np.linspace(0, env.path.length, 1000))

    axis_min_x = -30
    axis_max_x = 30
    axis_min_y = -10
    axis_max_y = 100
    axis_max = max(axis_max_x, axis_max_y)
    axis_min = min(axis_min_x, axis_min_y)

    X = np.arange(axis_min_x, axis_max_x, xstep)
    Y = np.arange(axis_min_y, axis_max_y, ystep)
    U, V = np.meshgrid(X, Y)

    i = 0
    for xidx, x in enumerate(X):
        for yidx, y in enumerate(Y):
            dist = np.sqrt((x-OBST_POSITION[0])**2 + (y-OBST_POSITION[1])**2)
            if (dist <= OBST_RADIUS):
                U[yidx][xidx] = 0
                V[yidx][xidx] = 0
                continue

            psi = np.pi/2
            rudder_angle = 1
            fil_psi = psi

            for _ in range(50):
                #env.reset()
                env.path = Path(waypoints)
                env.obstacles = obstacles
                env.vessel.reset([x, y, psi], [0, 0, 0])
                env.target_arclength = env.path.length
                obs, reward, done, info = env.step([0,0])
                #obs = env.observe()
                action, _states = agent.predict(obs, deterministic=True)
                last_rudder_angle = rudder_angle
                rudder_angle = action[1]
                thruster = action[0]

                psi -= rudder_angle * 0.1
                fil_psi = 0.8*fil_psi + 0.2*psi
            
            U[yidx][xidx] = thruster * np.cos(fil_psi)
            V[yidx][xidx] = thruster * np.sin(fil_psi)

            sys.stdout.write('Simulating behavior {:.2%} ({}/{})\r'.format((i+1)/(len(X)*len(Y)), (i+1), (len(X)*len(Y))))
            sys.stdout.flush()
            i += 1


    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    q = ax.quiver(X, Y, U, V)

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    ax.plot(path[0, :], path[1, :], dashes=[6, 2], linewidth=1.0, color='red', label=r'Path')
    ax.set_ylabel(r"North (m)")
    ax.set_xlabel(r"East (m)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y*10)))
    ax.set_xlim(axis_min_x-5, axis_max_x+5)
    ax.set_ylim(axis_min_y-5, axis_max_y+5)

    for obst in env.obstacles:
        circle = plt.Circle(
            obst.position,
            obst.radius,
            facecolor='tab:red',
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )
        obst = ax.add_patch(circle)
        obst.set_hatch('////')

    goal = plt.Circle(
        (path[0, -1], path[1, -1]),
        (axis_max - axis_min)/100,
        facecolor='black',
        linewidth=0.5,
        zorder=11
    )
    ax.add_patch(goal)
    ax.annotate("Goal", 
        xy=(path[0, -1] - (axis_max - axis_min)/15, path[1, -1]), 
        fontsize=12, ha="center", zorder=20, color='black',
    )

    #ax.legend()

    fig.savefig(os.path.join(fig_dir, fig_prefix + 'vector_field.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)

def test_report(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    class Struct(object): pass
    env = Struct()
    env.history = []
    env.episode = 1001
    env.config = gym_auv.DEFAULT_CONFIG
    for episode in range(1000):
        env.path = Struct()
        env.path.length = np.random.poisson(1000)
        progress = min(1, np.random.random() + np.random.random()*episode/1000)
        timesteps_baseline = env.path.length / (env.config["cruise_speed"] * env.config["t_step_size"])
        env.history.append({
            'collisions': np.random.poisson(0.1 + 7*(1-episode/1000)),
            'cross_track_error': np.random.gamma(10 - 5*episode/1000, 3),
            'progress': progress,
            'reached_goal': int(progress > 0.9),
            'reward': np.random.normal(-1000 + 2000*progress + episode**1.1, 2000),
            'timesteps': (np.random.random()+0.1)*np.random.poisson(timesteps_baseline + progress*500 + episode),
            'timesteps_baseline': timesteps_baseline,
            'surge': 10 + 3*np.random.normal(0, 2),
            'steer': np.random.normal(0, 25),
        })

    t = np.linspace(-400, 400, 1000)
    a = np.random.normal(0, 1)
    b = 100*np.random.random()
    c = 50*np.random.random() + 20
    d = 0.1*np.random.random()
    e = 100*np.random.random()
    f = 0.01*np.random.random() 
    g = 100*np.random.random()
    h = 0.01*np.random.random() 
    path_x = t + g*np.sin(h*t)
    path_y = a*t + b + e*np.sin(f*path_x)
    path_taken_x = path_x + e*np.sin(f*t)*np.cos(np.arctan2(path_y, path_x))
    path_taken_y = path_y + g*np.sin(h*t)*np.sin(np.arctan2(path_y, path_x))

    path = np.vstack((path_x, path_y))
    path_taken = np.vstack((path_taken_x, path_taken_y)).T

    env.last_episode = {
        'path': path,
        'path_taken': path_taken,
        'obstacles': []
    }
    env.obstacles = []
    for _ in range(np.random.poisson(10)):
        obst = Struct()
        s = int(1000*np.clip(np.random.random(), 0.1, 0.9))
        obst.position = np.array([
            path_x[s],
            path_y[s]
        ])
        obst.radius = np.random.poisson(30) 
        env.obstacles.append(obst)
        env.last_episode['obstacles'].append(('circle', [obst.position, obst.radius]))

    report(env, fig_dir)
    plot_trajectory(env, fig_dir)
