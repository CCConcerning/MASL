import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
from multiagent.core import Agent


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None, post_step_callback=None,
                 done_callback=None, other_callbacks=None, shared_viewer=True, discrete_action=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks场景回调
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.other_callbacks = other_callbacks
        self.post_step_callback = post_step_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward  hasattr() 函数用于判断对象是否包含对应的属性
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.collision = 0
        self.agent_colli = np.zeros((self.n, self.n))
        self.agent_distance = np.zeros((self.n, self.n))
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n, goal_n=None, restrict_move=False, return_phase_mask=False):
        obs_n = []
        reward_n = []
        env_reward_n = []
        done_n = []
        info_n = []  # {'n': []}
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # advance world state
        self.world.step()

        if restrict_move:
            # print("restrict move.............")
            for p in range(self.world.dim_p):
                for agent in self.agents:
                    if agent.adversary:
                        agent.state.p_pos[p] = np.sign(agent.state.p_pos[p]) * 1 if np.abs(
                            agent.state.p_pos[p]) > 1 else \
                            agent.state.p_pos[p]
                    else:
                        agent.state.p_pos[p] = np.sign(agent.state.p_pos[p]) if np.abs(agent.state.p_pos[p]) > 1 else \
                            agent.state.p_pos[p]
        if self.other_callbacks is not None:
            for call in self.other_callbacks:
                call(self.world)
        # record observation for each agent
        for aindex, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            if goal_n is not None:
                rew, ax_rew = self._get_reward(agent, goal_n)
                # reward_n.append(self._get_reward(agent, goal_n))
                reward_n.append(rew + ax_rew)
                env_reward_n.append(rew)
            else:
                reward_n.append(self._get_reward(agent))

            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))
        if self.post_step_callback is not None:
            self.post_step_callback(self.world)
            done = self._get_done(None)
            for agent in self.agents:
                done_n.append(done)

        reward = np.sum(reward_n)
        env_reward = np.sum(env_reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
            env_reward_n = [env_reward] * self.n

        self.collision += self.collision_sum()

        if goal_n is not None:
            return obs_n, reward_n, done_n, info_n, env_reward_n
        return obs_n, reward_n, done_n, info_n

    def get_agent_relative_distance(self):
        for aindex, agent in enumerate(self.world.agents):
            for oi, other in enumerate(self.world.agents):
                if other is agent: continue
                # 观察其他agents和自己的距离
                self.agent_distance[aindex][oi] = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
        return self.agent_distance

    def get_agent_to_lankmark_distance(self):
        dists = []
        for i, a in enumerate(self.world.agents):
            dists.append([np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in self.world.landmarks])
        return dists

    def occupied(self):
        occupied_landmarks = 0
        # print("...one step")
        for l in self.world.landmarks:
            # 算出每个agent离地标的欧式距离
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.world.agents]
            if min(dists) < 0.1:  # 0.03:#
                occupied_landmarks += 1
        return occupied_landmarks

    def hunt_occupied(self):
        occupied_landmarks = 0
        # print("...one step")
        for l in self.world.landmarks:
            # 算出每个agent离地标的欧式距离
            cout_agent = 0
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.world.agents]
            for d in dists:
                if d <= 0.1:
                    cout_agent += 1
            if cout_agent >= len(self.world.agents) / len(self.world.landmarks):
                occupied_landmarks += 1
        return occupied_landmarks

    def distance(self):
        min_dists = 0
        for l in self.world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in self.world.agents]
            min_dists += min(dists)
        return min_dists

    def hunt_distance(self):
        # min_dists = []
        min_dists = 0
        for a in self.world.agents:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in self.world.landmarks]
            min_dists += min(dists)
        return min_dists

    def get_collision(self):
        return self.collision

    def get_agent_collision(self):
        return self.agent_colli

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def collision_sum(self):
        count = []
        for aindex, agent in enumerate(self.world.agents):  # 算每一个agent的碰撞次数
            count.append(self.collision_count(agent, aindex))
        return np.sum(count)

    def collision_count(self, agent, aindex):
        count = 0
        if agent.collide:
            for ai, a in enumerate(self.world.agents):
                if a is agent: continue
                if self.is_collision(a, agent):
                    count += 1
                    self.agent_colli[aindex][ai] += 1
        return count

    def reset(self, with_goal=False, with_partner=False, return_phase_mask=False, return_info=False):
        # reset world
        self.collision = 0
        self.agent_colli = np.zeros((self.n, self.n))
        self.agent_distance = np.zeros((self.n, self.n))
        message = self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []

        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        if return_info:
            info_n = []
            for agent in self.agents:
                info_n.append(self._get_info(agent))
            return obs_n, info_n
        if with_goal:
            return obs_n, message
        elif with_partner:
            return obs_n, message
        elif return_phase_mask:
            phase_mask_n = [message for agent in self.agents]
            return obs_n, phase_mask_n
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return []  # {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent, goal_n=None):
        if self.reward_callback is None:
            return 0.0
        if goal_n is not None:
            return self.reward_callback(agent, self.world, goal_n)
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        if agent.movable == False:
            agent.action.u = [0, 0]
            agent.state.p_vel = [0, 0]
            action = []
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                elif 'obstacle' in entity.name:  # and entity.occupied:
                    geom.set_color(0.75, 0.75, 0.75, alpha=0.5)
                elif hasattr(entity, 'alive') and entity.alive == False:
                    geom.set_color(0.5, 0.5, 0.5, alpha=0.5)
                else:
                    geom.set_color(*entity.color)

                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent更新边界以代理为中心
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame在局部坐标系中创建受体场位置
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field圆形感受野
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin添加原点
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field网格感受野
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments一批多代理环境的矢量化包装器
# assumes all environments have the same observation and action space假设所有环境都具有相同的观察和行动空间
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
