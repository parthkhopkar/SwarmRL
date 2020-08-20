import os
import argparse
import json
import re

import tensorflow as tf
import numpy as np

from swarmnet import SwarmNet
from swarmnet.modules import MLP
from swarmnet.utils import load_model, one_hot, load_model_params
from tensorflow.python.ops.array_ops import newaxis

from ppo_agent import PPOAgent
from swarms.rl_extensions.envs import BoidSphereEnv2D

NDIM = 2
EDGE_TYPES = 4

NUM_BOIDS = 5
NUM_SPHERES = 1
NUM_GOALS = 1
DT = 0.3

ACTION_BOUND = 5.

T_MAX = 100


def system_edges(goals, obstacles, boids):
    """Edge types of the directed graph representing the influences between
    elements of the system.
        |    |Goal|Obst|Boid|
        |Goal| 0  | 0  | 1  |
        |Obst| 0  | 0  | 2  |
        |Boid| 0  | 0  | 3  |
    """
    particles = goals + obstacles + boids
    edges = np.zeros((particles, particles), dtype=int)

    up_to_goal = goals
    up_to_obs = up_to_goal + obstacles
    up_to_boids = up_to_obs + boids

    edges[0, up_to_obs:up_to_boids] = 1  # influence from goal to boid.
    edges[up_to_goal:up_to_obs, up_to_obs:up_to_boids] = 2  # influence from obstacle to boid.
    edges[up_to_obs:up_to_boids, up_to_obs:up_to_boids] = 3  # influence from boid to boid.

    np.fill_diagonal(edges, 0)
    return edges


def combine_env_states(agent_states, obstacle_states, goal_states):
    state = np.concatenate([goal_states, obstacle_states, agent_states], axis=0)
    return np.expand_dims(state, 0)  # Add time_seg_len dim to state for swarmnet


def combine_env_rewards(agent_reward, obstacle_reward, goal_reward):
    reward = np.concatenate([goal_reward, obstacle_reward, agent_reward], axis=-1)
    # assert reward.shape == (NUM_GOALS + NUM_SPHERES + NUM_BOIDS,)
    return reward


def get_swarmnet_actorcritic(params, log_dir=None):
    swarmnet, inputs = SwarmNet.build_model(
        NUM_GOALS+NUM_SPHERES+NUM_BOIDS, 2*NDIM, params, return_inputs=True)

    if log_dir:
        load_model(swarmnet, log_dir)

    # Action from SwarmNet
    actions = swarmnet.dense.output[:, :, NDIM:]

    # Value from SwarmNet
    encodings = swarmnet.graph_conv.output

    value_function = MLP([32, 1], activation=None, name=swarmnet.name_scope() +
                         '/value_function')
    values = value_function(encodings)  # shape [batch, num_nodes, 1]

    actorcritic = tf.keras.Model(inputs=inputs,
                                 outputs=[actions, values],
                                 name='SwarmnetActorcritic')

    return actorcritic


def main():
    swarmnet_params = load_model_params(ARGS.config)

    actorcritic = get_swarmnet_actorcritic(swarmnet_params, ARGS.log_dir)
    swarmnet_agent = PPOAgent(actorcritic, NDIM, ACTION_BOUND)

    env = BoidSphereEnv2D(NUM_BOIDS, NUM_SPHERES, NUM_GOALS, DT)
    edges = system_edges(NUM_GOALS, NUM_SPHERES, NUM_BOIDS)
    edge_types = one_hot(edges, EDGE_TYPES)

    reward_all_episodes = []
    for episode in range(ARGS.epochs):
        state = env.reset()
        state = combine_env_states(*state)
        reward_episode = 0
        for t in range(T_MAX):
            action = swarmnet_agent.act(
                [state[np.newaxis, ...], edge_types[np.newaxis, ...]])

            next_state, reward, done = env.step(action)
            reward = combine_env_rewards(*reward)

            swarmnet_agent.store_transition([state, edge_types], action, reward)

            state = combine_env_states(*next_state)
            reward_episode += np.sum(reward)

            if len(swarmnet_agent.rollout_buffer) >= ARGS.batch_size or done:
                swarmnet_agent.finish_rollout(
                    [state[np.newaxis, ...], edge_types[np.newaxis, ...]], done)
                swarmnet_agent.update()
                print(f'  Break at Step {t}')
                break

        reward_all_episodes.append(reward_episode)
        print(
            f'\r Episode {episode} | Reward {reward_episode:.2f} | Avg. Reward {np.mean(reward_all_episodes[-20:])}', end='')
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()
