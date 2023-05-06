import core.common.tf_util as U
import numpy as np
import tensorflow as tf
from core import AgentTrainer
from core.common.distributions import make_pdtype
from core.trainer.replay_buffer import ReplayBuffer
from core.common.misc import onehot_from_logits, categorical_sample
import tensorflow.contrib.layers as layers


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, scale_ob, act_space_n, p_index, p_func, q_func, batch, optimizer, bw_optimizer,
            grad_norm_clipping=None,
            local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        obs_ph_n = make_obs_ph_n
        scale_obs_ph_n = scale_ob
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        select = tf.placeholder(tf.float32, [None, None], name="select")
        s_act = tf.placeholder(tf.float32, [None, 1, np.shape(act_ph_n[0])[1]], name="s_act")
        s_ob = tf.placeholder(tf.float32, [None, 1, np.shape(obs_ph_n[0])[1]], name="s_ob")

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))  # 网络参数

        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()
        b_reg = tf.reduce_mean(tf.square(act_pd.flatparam() - act_ph_n[p_index]))
        b_loss = b_reg

        b_optimize_expr = U.minimize_and_clip(bw_optimizer, b_loss, p_func_vars, grad_norm_clipping)

        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        q_input_self = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        other_obs = list(np.delete(scale_obs_ph_n, p_index, axis=0))
        other_act = list(np.delete(act_input_n, p_index, axis=0))
        q_input_other = tf.concat(other_obs + other_act, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)

        q = q_func(q_input_self, q_input_other, act_input_n[p_index], 1, scope="q_func", reuse=True,
                   num_units=num_units)[:, 0]

        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        train = U.function(inputs=scale_obs_ph_n + obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        b_train = U.function(inputs=[obs_ph_n[p_index]] + [act_ph_n[p_index]], outputs=b_loss,
                             updates=[b_optimize_expr])
        pl = U.function(scale_obs_ph_n + obs_ph_n + act_ph_n + [s_act] + [s_ob] + [select], pg_loss)
        pr = U.function(scale_obs_ph_n + obs_ph_n + act_ph_n, p_reg * 1e-3)
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        bob = U.function(inputs=[obs_ph_n[p_index]], outputs=p_input)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()

        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, b_train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_input_self = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)

        other_obs = list(np.delete(obs_ph_n, q_index, axis=0))
        other_act = list(np.delete(act_ph_n, q_index, axis=0))

        q_input_other = tf.concat(other_obs + other_act, 1)

        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input_self, q_input_other, act_ph_n[q_index], 1, scope="q_func", num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        # q_input=tf.concat(obs_ph_n + act_ph_n, 1)
        target_q = q_func(q_input_self, q_input_other, act_ph_n[q_index], 1, scope="target_q_func",
                          num_units=num_units)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


def compute_relevancy(n, agent_index, obs_n, obs, select_num):
    other_obs = list(np.delete(obs_n, agent_index, axis=0))
    outputs = np.matmul(np.expand_dims(obs, 1), np.transpose(other_obs, [1, 2, 0]))
    out_att = outputs / np.sqrt(int(other_obs[0].shape[1]))
    weight = np.exp(out_att) / np.sum(np.exp(out_att), axis=2, keepdims=True)
    mean_weight = np.mean(weight, axis=0)
    ind = [i for i in range(n) if i != agent_index]
    index = np.random.choice(a=ind, size=select_num, replace=False, p=mean_weight[0])  # 这个
    return index


class MASLAgentTrainer(AgentTrainer):
    def __init__(self, name, model, qmodel, obs_shape_n, act_space_n, agent_index, args, local_q_func=False,
                 logger=None):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.look = [i for i in range(len(obs_shape_n)) if i != agent_index]
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=qmodel,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        self.act, self.p_train, self.b_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            scale_ob=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=qmodel,
            batch=args.batch_size,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            bw_optimizer=tf.train.AdamOptimizer(learning_rate=args.bw_lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6, random_seed=args.random_seed)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.obs_next_mean = None
        self.obs_next_std = None
        self.obs_mean = None
        self.obs_std = None
        self.actions_mean = None
        self.actions_std = None

    def max_len(self):
        return self.max_replay_buffer_len

    def len_buffer(self):
        return len(self.replay_buffer)

    def action(self, obs):
        return self.act(obs[None])[0]

    def Q_value(self, obs, act):
        return self.q_debug['q_values'](*(obs + act))

    def experience(self, obs, act, rew, new_obs, done, terminal):

        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def imi_update(self, obs_n, act_n, t, logger):
        loss = self.b_train(*(obs_n + act_n))
        logger.add_scalars("agent%i/imi_losses" % self.agent_index,
                           {"imi_loss": loss}, t)

    def update(self, agents, t, episode, select_num, logger):
        if len(self.replay_buffer) < 0:
            return
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        rew_n = []
        select_act = []
        index = self.replay_sample_index
       
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            rew_n.append(rew)

        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # select K agents by computing relevancy
        index = compute_relevancy(self.n, self.agent_index, obs_n, obs, select_num)
        # train q network
        num_sample = 1
        target_q = 0.0
        target_q_next = 0
        for i in range(num_sample):
            target_act_next_n = act_n + []
            select_obs_next = obs_next_n + []
            select_obs = obs_n + []
            select_act = act_n + []
            for s in range(self.n):
                if s in index or s == self.agent_index:
                    target_act_next_n[s] = self.p_debug['target_act'](obs_next_n[s])
                    select_obs_next[s] = obs_next_n[s]
                    select_obs[s] = obs_n[s]
                    select_act[s] = act_n[s]
                else:
                    target_act_next_n[s] = np.zeros(act_n[0].shape)
                    select_obs_next[s] = np.zeros(obs_next_n[0].shape)
                    select_obs[s] = np.zeros(obs_n[0].shape)
                    select_act[s] = np.zeros(act_n[0].shape)

            target_q_next = self.q_debug['target_q_values'](*(select_obs_next + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        q_loss = self.q_train(*(select_obs + select_act + [target_q]))
        p_loss = self.p_train(*(select_obs + obs_n + select_act))
        # print('p_loss', p_loss)
        self.p_update()
        self.q_update()
        logger.add_scalars("agent%i/losses" % self.agent_index,
                           {"vf_loss": q_loss,
                            "pol_loss": p_loss,
                            },
                           t / 50)

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
