import argparse
import os

import numpy as np
import tensorflow as tf
import time
import random
import pickle
import core.common.tf_util as U
from core.trainer.masl_trainer import MASLAgentTrainer
from core.trainer.bw import bwmodel
import tensorflow.contrib.layers as layers
from tensorboardX import SummaryWriter

'''
Ensure to install the multiagent particle envs first.
Then, run  "python train.py --scenario hunt" for rover exploration
           "python train.py --scenario simple_spread" for resource collection
'''
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="hunt", help="name of the scenario script")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=111)
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--bw_lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    
    parser.add_argument("--exp-name", type=str, default="MASL", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    # Buffer Params for BW model
    parser.add_argument('--capacity', type=int, default=10000, help='the capacity of the replay buffer')
    parser.add_argument('--per-weight', action='store_true', default=False, help='whether to enable PER')
    parser.add_argument('--sil-alpha', type=float, default=0.6, help='the exponent for PER')
    parser.add_argument('--sil-beta', type=float, default=0.4, help='sil beta')
    # BW Specific config
    parser.add_argument('--bw', action='store_true', default=False, help='Enable BW Model')
    parser.add_argument('--k-states', type=int, default=1000,
                        help='Number of top value states to train Backtracking Model on')
    parser.add_argument('--num-states', type=int, default=1, help='Number of high value state to actually backtrack on')
    parser.add_argument('--trace-size', type=int, default=3,
                        help='Number of steps to backtrack on for a given high value state ie length of trajectory')
    parser.add_argument('--consistency', action='store_true', help='For consistency bw forward and backward model')
    parser.add_argument('--logclip', type=float, default=4.0, help='Clipping for log Normal')
    parser.add_argument('--n-ac', type=int, default=250, help='Number of a2c updates after which to do bw update')
    parser.add_argument('--n-bw', type=int, default=1, help='Number of bw updates to do per n-a2c updates')
    parser.add_argument('--n-imi', type=int, default=1, help='Number of imitation updates to do per n-a2c updates')
    parser.add_argument('--entropy-coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='maex norm of gradients (default: 0.5)')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--ratio', type=float, default=0.4)
    return parser.parse_args()


def act_mlp_model(input, num_outputs, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def state_mlp_model(input, ob, num_outputs, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def actor(input, num_outputs, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def critic(input_self, input_other, self_sa, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out1 = layers.fully_connected(input_self, num_outputs=128, activation_fn=tf.nn.relu)
        out1 = layers.fully_connected(out1, num_outputs=96, activation_fn=tf.nn.relu)
        out2 = input_other
        out2 = layers.fully_connected(out2, num_outputs=96, activation_fn=tf.nn.relu)
        out = tf.concat([out1, out2], 1)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = actor
    critic_model = critic
    trainer = MASLAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, critic_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, critic_model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def get_bw_trainers(env, trainers, obs_shape_n, arglist):
    act_model = act_mlp_model
    state_model = state_mlp_model
    trainer = bwmodel
    bw_trainers = trainer("bw", act_model, state_model, trainers, obs_shape_n, env.action_space, arglist)
    return bw_trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        np.random.seed(arglist.random_seed)
        tf.set_random_seed(arglist.random_seed)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)

        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        if arglist.bw:
            bw_trainers = get_bw_trainers(env, trainers, obs_shape_n, arglist)

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
        os.makedirs(arglist.plots_dir, exist_ok=True)
        log_dir = arglist.save_dir + "/logs/"
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(str(log_dir))
        episode_rewards = [0.0]
        last_reward = [0.0]
        final_ep_rewards = []  # sum of rewards for training curve
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        distance = []
        collision = []
        occupied = []
        update = 0
        imitation = False
        retrogession = False
        identified_id = -1
        max_reward = 0

        print(env.n, 'Starting Training...')
        while True:
            if not retrogession:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            if arglist.bw:
                select_agents = []
                if retrogession is False and update >= 2 and np.max(rew_n) >= max_reward:
                    update = 0
                    retrogession = True
                    select_obs = new_obs_n[:]  # record possible high-value states
                    record_reward = rew_n[:]
                    if np.max(record_reward) > max_reward:  # update max reward
                        max_reward = np.max(record_reward)
                    record = np.sum(record_reward)

                    speed = []
                    for i in range(env.n):
                        speed.append([new_obs_n[i][0], new_obs_n[i][1]])
                if retrogession:
                    if np.abs(record) > np.abs(np.sum(rew_n)) or np.sum(rew_n) <= 0:
                        select_agents.append(identified_id)
                        record = np.sum(rew_n)
                        bw_trainers.add_high_value_state(select_obs[identified_id])
                    else:
                        imitation = False
                    if np.max(
                            rew_n) < 0 or identified_id == env.n - 1:  # All the high-value states have been identified
                        identified_id = -1
                        retrogession = False
                    else:
                        identified_id += 1

                bw_trainers.step(terminal,
                                 select_agents,
                                 obs_n,
                                 action_n,
                                 rew_n,
                                 done_n,
                                 new_obs_n)

            # collect experience
            if not retrogession:
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

            obs_n = new_obs_n

            episode_rewards[-1] += rew_n[0]  

            if done or terminal:  
                occupied.append(env.hunt_occupied())
                distance.append(env.hunt_distance())
                collision.append(env.get_collision())

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)  
                retrogession = False
                identified_id = -1

            if retrogession:
                for i in range(env.n):
                    speed.append([new_obs_n[i][0], new_obs_n[i][1]])
                if identified_id == 0:
                    v = 3.5 if i == 0 else 1.5
                    return np.array([0, 0, speed[i][0] * v, 0, speed[i][1] * v])
                else:
                    if i == identified_id:
                        v = 2
                    elif i == identified_id - 1:
                        v = -1.5
                    else:
                        v = 0
                    action_n[i] = np.array([0, 0, speed[i][0] * v, 0, speed[i][1] * v])

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if train_step % 50 == 0:  # only update every 50 steps
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step, len(episode_rewards), 5, logger)
                update += 1

            if arglist.bw and train_step % 200 == 0:  
                imitation = True
            if arglist.bw and bw_trainers.num_steps() > 100 and imitation: 
                for _ in range(arglist.n_bw):
                    bw_trainers.train_bw_model(logger)

                for _ in range(arglist.n_imi):
                    bw_trainers.train_imitation(logger)
                imitation = False

            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if np.mean(occupied[-arglist.save_rate:]) >= 1.5:
                    arglist.bw = False
                if num_adversaries == 0:
                    print(
                        "steps: {}, episodes: {},occupied:{}, mean distance: {:.2}, sum of reward: {}, time: {}".format(
                            train_step, len(episode_rewards),
                            np.mean(occupied[-arglist.save_rate:]),
                            np.mean(episode_rewards[-arglist.save_rate:]),
                            np.mean(distance[-arglist.save_rate:]),
                            np.mean(last_reward[-arglist.save_rate:]),
                            round(time.time() - t_start, 3)))

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))

            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
