import os
import argparse
import datetime

import tensorflow as tf
import numpy as np

from swarmnet import SwarmNet
from swarmnet.modules import MLP
from swarmnet.utils import save_model, load_model, one_hot, load_model_params
from tensorflow.python.keras.backend import update

import utils
from ppo_agent import ACTOR_UPDATE_STEPS, PPOAgent
from gym_pybullet_drones.envs.multi_agent_rl.GoalAviary import GoalAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import \
    ActionType
from gym_pybullet_drones.utils.utils import sync

Z = 0.5
NDIM = 2
EDGE_TYPES = 4

MIN_NUM_DRONES = 1
MAX_NUM_DRONES = 5
MIN_NUM_OBSTACLES = 1
MAX_NUM_OBSTACLES = 1
NUM_GOALS = 1
MAX_NUM_NODES = MAX_NUM_DRONES + MAX_NUM_OBSTACLES + NUM_GOALS
DT = 0.3

BOID_SIZE = 2
SPHERE_SIZE = 7
NOISE = 0.0

ACTION_BOUND = 5. * DT

ROLLOUT_STEPS = 8
TRAIN_FREQUENCY = 4096
T_MAX = 60


def set_init_weights(model):
    init_weights = [weights / 10 for weights in model.get_weights()]
    model.set_weights(init_weights)


def get_swarmnet_actorcritic(params, log_dir):
    swarmnet, inputs = SwarmNet.build_model(
        MAX_NUM_NODES, 2 * NDIM, params, return_inputs=True)

    # Action from SwarmNetW
    actions = swarmnet.out_layer.output[:, :, NDIM:]

    # Value from SwarmNet
    encodings = swarmnet.graph_conv.output[:, :, :]

    value_function = MLP([64, 64, 1], activation=None, name='value_function')
    values = value_function(encodings)  # shape [batch, NUM_GOALS+MAX_NUM_SPHERES+MAX_NUM_BOIDS, 1]

    actorcritic = tf.keras.Model(inputs=inputs,
                                 outputs=[actions, values],
                                 name='SwarmnetActorcritic')

    # Register non-overlapping `actor` and `value_function` layers for fine control
    # over trainable_variables
    actorcritic.encoding = swarmnet.graph_conv
    actorcritic.actor = swarmnet.out_layer
    actorcritic.critic = value_function

    set_init_weights(actorcritic)
    load_model(swarmnet, log_dir)

    return actorcritic


def train(agent, value_only=False):
    # Fix goal-agent edge function
    goal_edge = agent.model.encoding.edge_encoder.edge_encoders[0]
    if ARGS.mode > 0:
        goal_edge.trainable = False

    reward_all_episodes = []
    ts = []
    step = 0
    initial_positions = [[-1,1,Z], [-1,1.5,Z], [-1,0,Z],[-1,-0.5,Z],[-1,-1,Z]]
    goal_x, goal_y = 3.,3.
    goal_pos = [goal_x, goal_y, 0.05]
    obstacle_x, obstacle_y = 1.5, 1.5
    obstacle_pos = [(obstacle_x, obstacle_y, Z)]
    obstacle_present = True
    static_entities = 1 + (1 if obstacle_present else 0)
    # Initialize num_boids and num_spheres.
    num_drones = MAX_NUM_DRONES  # np.random.randint(MIN_NUM_DRONES, MAX_NUM_DRONES + 1)
    num_obstacles = MAX_NUM_OBSTACLES  # np.random.randint(MIN_NUM_OBSTACLES, MAX_NUM_OBSTACLES + 1)
    env = GoalAviary(gui=False, 
                    record=False,
                    num_drones=num_drones,
                    act=ActionType.PID,
                    initial_xyzs=np.array(initial_positions),
                    aggregate_phy_steps=int(5),
                    goal_pos=goal_pos,
                    obstacle_pos=obstacle_pos,
                    noise=NOISE
                    )
    for episode in range(ARGS.epochs):
        # Create mask.
        mask = utils.get_mask(num_drones, MAX_NUM_NODES)
        # Form edges as part of inputs to swarmnet.
        # TODO: Edges are now returned by the environment
        edges = utils.system_edges(NUM_GOALS, num_drones, num_obstacles)
        edge_types = one_hot(edges, EDGE_TYPES)
        padded_edge_types = utils.pad_data(edge_types, MAX_NUM_NODES, dims=[0, 1])

        # TODO: Create env(s) once outside the loop and reset it for now 
        # env = BoidSphereEnv2D(num_boids, num_spheres, NUM_GOALS, DT,
        #                       boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE)
        state = env.reset()  # Received observations for each agent as a nested dictionary {i: {'nodes': ... , 'edges': ...}}
        for agent_idx in range(num_drones):
            state[agent_idx]['nodes'] = np.expand_dims(state[agent_idx]['nodes'],0)  # Insert time step dimension

        # state = utils.combine_env_states(*state)  # Shape [1, NUM_GOALS+num_spahers+num_boids, 4]
        action = {i:np.array([0.,0.,0.]) for i in range(num_drones)}
        reward_episode = 0
        for t in range(T_MAX):
            # Lists for storing the states of all agents
            padded_states = []
            padded_actions = []
            padded_log_probs = []
            padded_next_states = []
            padded_rewards = []
            for agent_idx in range(num_drones):
                padded_state = utils.pad_data(state[agent_idx]['nodes'], MAX_NUM_NODES, [1])
                padded_action, padded_log_prob = agent.act([padded_state, padded_edge_types], mask, training=True)
                # Store in lists
                padded_states.append(padded_state)
                padded_actions.append(padded_actions)
                padded_log_probs.append(padded_log_prob)

                print(f'Agent {agent_idx} action', padded_action, padded_action.shape)
                # Fill in action dict
                # Ignore "actions" from goals and obstacles.
                action[agent_idx][:2] = padded_action[-num_drones+agent_idx]

            # Step in the environment
            next_state, reward, done, info = env.step(action)

            for agent_idx in range(num_drones):
                next_state[agent_idx]['nodes'] = np.expand_dims(next_state[agent_idx]['nodes'],0)  # Insert time step dimension
                padded_next_state = utils.pad_data(next_state[agent_idx]['nodes'], MAX_NUM_NODES, [1])
                padded_reward = utils.pad_data(reward[agent_idx], MAX_NUM_NODES, [0])
                # Store in lists
                padded_next_states.append(padded_next_state)
                padded_rewards.append(padded_reward)

            for agent_idx in range(num_drones):
                agent.store_transition([padded_states[agent_idx], padded_edge_types], padded_actions[agent_idx], padded_rewards[agent_idx],
                                   padded_log_probs[agent_idx], [padded_next_states[agent_idx], padded_edge_types], done, mask)

            state = next_state
            reward_episode += np.sum(reward)

            step += 1
            if done or (t == T_MAX - 1):
                agent.finish_rollout([padded_state, padded_edge_types], done, mask)

            if step % TRAIN_FREQUENCY == 0:
                agent.update(ARGS.batch_size, actor_steps=int(not value_only))
            if done:
                break

        env.close()

        ts.append(t)
        reward_all_episodes.append(reward_episode)

        # Log to tensorboard
        with agent.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', reward_episode, step=episode)
            tf.summary.scalar('Terminal Timestep', t, step=episode)

        print(f'\r Episode {episode} | Reward {reward_episode:8.2f} | ' +
              f'Avg. R {np.mean(reward_all_episodes[-100:]):8.2f} | Avg. End t = {np.mean(ts[-100:]):3.0f}',
              end='')
        if (episode + 1) % 1000 == 0:
            print('')
            # Hack for preserving the order or weights while saving
            goal_edge.trainable = True
            save_model(agent.model, ARGS.log_dir + '/rl')
            save_model(agent.model, ARGS.log_dir + f'/rl_{episode}')
            if ARGS.mode > 0:
                goal_edge.trainable = False
            np.save(ARGS.log_dir + '/rl/train_rewards.npy', reward_all_episodes)
            np.save(ARGS.log_dir + '/rl/terminal_ts.npy', ts)

    goal_edge.trainable = True
    save_model(agent.model, ARGS.log_dir + '/rl')


def test(agent):
    # Form edges as part of inputs to swarmnet.
    num_boids = np.random.randint(MIN_NUM_BOIDS, MAX_NUM_BOIDS + 1)
    num_spheres = np.random.randint(MIN_NUM_SPHERES, MAX_NUM_SPHERES + 1)

    env = BoidSphereEnv2D(num_boids, num_spheres, NUM_GOALS, DT,
                          boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE)

    edges = utils.system_edges(NUM_GOALS, num_spheres, num_boids)
    edge_types = one_hot(edges, EDGE_TYPES)

    state = env.reset(ARGS.seed)
    state = utils.combine_env_states(*state)
    reward_sequence = []
    trajectory = [state]

    if ARGS.gif:
        import imageio
        frames = []

    for t in range(ARGS.steps):
        action, _ = agent.act([state, edge_types])
        value = agent.value([state, edge_types])

        print(f'Step {t}')
        print('Action', action)
        # print(test_out)

        # Ignore "actions" from goals and obstacles.
        next_state, reward, done = env.step(action)

        if ARGS.gif:
            frames.append(env.render())
        else:
            env.render()
        # reward = combine_env_rewards(*reward)

        state = utils.combine_env_states(*next_state)

        reward_sequence.append(reward)
        trajectory.append(state)

        if done:
            break

    print(f' Final Reward {np.sum(reward_sequence)} | End t = {t}')
    np.save(os.path.join(ARGS.log_dir, 'test_trajectory.npy'), trajectory)
    np.save(os.path.join(ARGS.log_dir, 'reward_sequence.npy'), reward_sequence)
    if ARGS.gif:
        print('Saving GIF...')
        imageio.mimsave(os.path.join(ARGS.log_dir, ARGS.gif + '.gif'), frames, fps=6)


def main():
    # Tensorboard logging setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(
        ARGS.log_dir + '/' + current_time) if ARGS.train or ARGS.pretrain else None

    swarmnet_params = load_model_params(ARGS.config)

    actorcritic = get_swarmnet_actorcritic(swarmnet_params, ARGS.log_dir)
    # NOTE: lock node_updater layer and final dense layer.
    if ARGS.mode == 2:
        actorcritic.encoding.node_decoder.trainable = False
        actorcritic.actor.trainable = False

    # Load weights trained from RL.
    rl_log = os.path.join(ARGS.log_dir, 'rl')
    if os.path.exists(rl_log):
        load_model(actorcritic, rl_log)

    swarmnet_agent = PPOAgent(actorcritic, NDIM,
                              action_bound=None,
                              rollout_steps=ROLLOUT_STEPS,
                              memory_capacity=4096,
                              summary_writer=summary_writer,
                              mode=ARGS.mode)

    if ARGS.pretrain:
        train(swarmnet_agent, value_only=True)
    elif ARGS.train:
        train(swarmnet_agent)
    elif ARGS.test:
        test(swarmnet_agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training steps')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='batch size')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='turn on pretraining of value function')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    parser.add_argument('--seed', type=int, default=1337,
                        help='set random seed')
    parser.add_argument("--gif", type=str, default=None,
                        help="store output as gif with the given filename")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=T_MAX,
                        help='max timestep per episode')
    ARGS = parser.parse_args()

    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    utils.set_seed(ARGS.seed)
    main()
