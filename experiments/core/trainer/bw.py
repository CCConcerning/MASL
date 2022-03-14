import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer, High_Value_ReplayBuffer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def zero_mean_unit_std(dataX):
    mean_x = np.mean(dataX, axis=0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis=0)
    np.seterr(divide='ignore', invalid='ignore')
    dataX = np.nan_to_num(dataX / std_x)
    return dataX, mean_x, std_x


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, obs_deta_n, act_space_n, p_func, q_func, optimizer, grad_norm_clipping=None, num_units=64,
            scope="bw_trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        deta_state_pdtype_n = [make_pdtype(obs_deta) for obs_deta in obs_deta_n]
        next_obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[0].sample_placeholder([None], name="bw_action")]
        deta_state_ph_n = [deta_state_pdtype_n[0].sample_placeholder([None], name="deta_state")]
        p_input = next_obs_ph_n[0]
        p = p_func(p_input, int(act_pdtype_n[0].param_shape()[0]), scope="bw_action_func",
                   num_units=num_units)  # Actgen network output
        p_func_vars = U.scope_vars(U.absolute_scope_name("bw_action_func"))

        q_input = tf.concat(next_obs_ph_n + act_ph_n, 1)
        q = q_func(q_input, act_ph_n[0], int(deta_state_pdtype_n[0].param_shape()[0]), scope="bw_state_func",
                   num_units=num_units)
        q_func_vars = U.scope_vars(U.absolute_scope_name("bw_state_func"))
        bw_vars = p_func_vars + q_func_vars
        # wrap parameters in distribution
        act_pd = act_pdtype_n[0].pdfromflat(p)
        act_sample = act_pd.sample()  # sample action

        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam() - act_ph_n[0]))

        q_reg = tf.reduce_mean(tf.square(deta_state_ph_n - q))

        loss = p_reg + q_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, bw_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=next_obs_ph_n + act_ph_n + deta_state_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=next_obs_ph_n, outputs=act_sample)
        state_loss = U.function(inputs=next_obs_ph_n + act_ph_n + deta_state_ph_n, outputs=q_reg)
        act_loss = U.function(inputs=next_obs_ph_n + act_ph_n, outputs=tf.reduce_mean(p_reg))
        p_values = U.function(next_obs_ph_n, p)

        deta_state = U.function(inputs=next_obs_ph_n + act_ph_n, outputs=q)

        return act, deta_state, train


class bwmodel():
    def __init__(self, name, act_model, state_model, maddpg, obs_shape_n, act_space_n, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.args = args

        self.action_shape = [act_space_n[0]]
        self.obs_shape = [obs_shape_n[0]]
        self.maddpg = maddpg
        obs_ph_n = []
        obs_ph_n.append(U.BatchInput(obs_shape_n[0], name="next_observation").get())
        self.act, self.deta_state, self.train = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            obs_deta_n=self.obs_shape,
            act_space_n=self.action_shape,
            p_func=act_model,
            q_func=state_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.bw_lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units
        )

        # Create experience buffer
        self.high_value_buffer = High_Value_ReplayBuffer(1000, random_seed=args.random_seed)
        self.buffer = ReplayBuffer(self.args.capacity, random_seed=args.random_seed)
        self.replay_sample_index = None
        # some other parameters...
        self.running_episodes = [[] for _ in range(self.n)]
        self.total_steps = []
        self.total_rewards = []
        # Set the mean, stds. All numpy stuff
        self.obs_delta_mean = None
        self.obs_delta_std = None
        self.obs_next_mean = None
        self.obs_next_std = None
        self.obs_mean = None
        self.obs_std = None
        self.actions_mean = None
        self.actions_std = None
        self.select = [0 for _ in range(self.n)]
        self.niter = 0
        self.imi_train_step = 0

    def train_bw_model(self, logger):
        obs, actions, returns, obs_next_unnormalized, done = self.sample_batch_noper(self.args.capacity)

        batch_size = min(self.args.k_states, len(self.buffer))
        if obs is not None and obs_next_unnormalized is not None:
            sorted_indices = returns.reshape(-1).argsort()[-batch_size:][::-1]
            obs = obs[sorted_indices.tolist()]
            obs_next_unnormalized = obs_next_unnormalized[sorted_indices.tolist()]
            actions = actions[sorted_indices.tolist()]
            returns = returns[sorted_indices.tolist()]
            obs_delta, self.obs_delta_mean, self.obs_delta_std = zero_mean_unit_std(obs - obs_next_unnormalized)
            actions, self.actions_mean, self.actions_std = zero_mean_unit_std(actions)
            obs, self.obs_mean, self.obs_std = zero_mean_unit_std(obs)
            obs_next, self.obs_next_mean, self.obs_next_std = zero_mean_unit_std(obs_next_unnormalized)
            avg_loss = 0

            for _ in range(self.args.epoch):
                total_loss = self.train(*([obs_next] + [actions] + [obs_delta]))
                avg_loss += total_loss
            avg_loss /= self.args.epoch
            self.niter += 1

            if logger is not None:
                logger.add_scalars("bw_losses",
                                   {"bw_loss": avg_loss},
                                   self.niter)

    def get_high_value_buffer_len(self):
        return self.high_value_buffer.__len__()

    def train_imitation(self, logger):
        num_states = 5
        index = self.high_value_buffer.make_index(num_states)
        states = self.high_value_buffer.sample_index(index)

        if 0 in self.obs_next_std:
            return 0.0
        if states is not None and not np.isnan(np.sum(self.obs_next_std)):
            np.seterr(divide='ignore', invalid='ignore')
            states_normalized = np.nan_to_num((states - self.obs_next_mean) / self.obs_next_std)
            if np.isnan(np.sum(states_normalized)) or np.isinf(np.sum(states_normalized)):
                return 0.0
            mb_actions, mb_states_prev = [], []
            # Sample the Traces
            size = 0
            for step in range(self.args.trace_size):
                if np.isnan(np.sum(states_normalized)) or np.isinf(np.sum(states_normalized)):
                    return 0
                actions = self.act(states_normalized)
                if np.isnan(np.sum(actions)) or np.isinf(np.sum(actions)):
                    return 0

                deta_state = self.deta_state(*([states_normalized] + [actions]))
                actions = actions * self.actions_std + self.actions_mean
                deta_state = deta_state * self.obs_delta_std + self.obs_delta_mean
                states_prev = states + deta_state
                states_next = states_prev
                states_normalized = np.nan_to_num((states_next - self.obs_mean) / self.obs_next_std)
                if not np.isnan(np.sum(states_prev)):
                    if size == 0:
                        mba = actions
                        mbs = states_prev
                    else:
                        mba = np.vstack((mba, actions))
                        mbs = np.vstack((mbs, states_prev))
                    size += 1
                mb_actions = [mba]
                mb_states_prev = [mbs]

            for agent in self.maddpg:
                loss = agent.imi_update(mb_states_prev, mb_actions, self.imi_train_step, logger)
                self.imi_train_step += 1
            return loss
        else:
            return 0.0

    def add_high_value_state(self, obs):
        self.high_value_buffer.add(obs)

    def step(self, terminal, select_agents, obs, actions, rewards, dones, obs_next):
        """
        Add the batch information into the Buffers
        """
        if select_agents is not None:
            for i in select_agents:
                self.select[i] += 1

        for n in range(self.n):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n], obs_next[n], dones[n]])
        if terminal:
            for n, select in enumerate(self.select):
                if select > 0 and rewards[n] > 0:
                    self.update_buffer(self.running_episodes[n])
                # Clear the episode buffer
                self.running_episodes[n] = []
                self.select[n] = 0
        if len(self.buffer) >= self.args.capacity - 1:
            # Buffer is full
            if self.buffer._next_idx > int(self.args.ratio * self.args.capacity):
                # Limit reached. Sort and 0
                self.buffer._storage.sort(key=lambda x: x[2])
                self.buffer._next_idx = self.buffer._next_idx % int(self.args.ratio * self.args.capacity)

    def update_buffer(self, trajectory):
        """
        Update buffer. Add single episode to PER Buffer and update stuff
        """
        self.add_episode(trajectory)
        self.total_steps.append(len(trajectory))
        self.total_rewards.append(np.sum([x[2] for x in trajectory]))
        while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
            self.total_steps.pop(0)
            self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        """
        Add single episode to PER Buffer
        """
        obs = []
        actions = []
        rewards = []
        dones = []
        obs_next = []
        for (ob, action, reward, ob_next, done) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            if ob_next is not None:
                obs_next.append(ob_next)
            else:
                obs_next.append(None)
            actions.append(action)
            # rewards.append(np.sign(reward))
            rewards.append(reward)
            dones.append(False)
        # Put done at end of trajectory
        dones[len(dones) - 1] = True
        returns = discount_with_dones(rewards, dones, 0.95)
        for (ob, action, R, ob_next, done) in list(zip(obs, actions, returns, obs_next, dones)):
            self.buffer.add(ob, action, R, ob_next, float(done))

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None, None

    def sample_batch_noper(self, batch_size):
        # 100
        if len(self.buffer) >= 100:
            batch_size = min(batch_size, len(self.buffer))
            # print(batch_size)
            return self.buffer.sample(batch_size)
        else:
            return None, None, None, None, None
