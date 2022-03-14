import numpy as np
from multiagent.core_prey_1 import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        num_landmarks = 5
        num_obstacle = 0
        world.collaborative = False
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.i = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        world.obstacle = [Landmark() for i in range(num_obstacle)]
        for i, obstacle in enumerate(world.obstacle):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.08
            obstacle.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, obstacle in enumerate(world.obstacle):
            obstacle.color = np.array([0.55, 0.55, 0.55])
            obstacle.state.p_pos = np.array([0.0, 0.0])
            obstacle.state.p_vel = np.zeros(world.dim_p)
        # set random initial states

        for i, agent in enumerate(world.agents):
            # numpy.random.uniform(low,high,size)从一个均匀分布[low,high)中随机采样 size: 输出样本数目
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
        p = []
        p.append(np.array([0.273221902, -0.6659316]))
        p.append(np.array([-0.74646218, -0.01075534]))
        p.append(np.array([0.25743576, 0.77392895]))
        p.append(np.array([0.5, -0.2]))
        p.append(np.array([-0.6, -0.6]))

        p.append(np.array([-0.46937189, 0.74963775]))
        p.append(np.array([0.54989629, 0.34437029]))
        p.append(np.array([0., -0.]))

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = p[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        # print(agent1.state.p_pos)
        # print(agent2.state.p_pos)
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # 代理商根据与每个地标的最小代理距离进行奖励，对冲突进行处罚
        rew = 0
        occupied_landmarks = 0
        # print("...one step")
        for l in world.landmarks:
            # 算出每个agent离地标的欧式距离
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.08:
                occupied_landmarks += 1

        rew += 1 * (occupied_landmarks ** 2)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.1
        return rew

    # individual reward
    def reward_individual(self, agent, world):
        rew = 0
        occupied_landmarks = 0

        for l in world.landmarks:
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            if dists < 0.08:
                occupied_landmarks += 1
        rew += occupied_landmarks * 5

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.1
        return rew

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        other_pos = []
        aindex = []
        for ai, other in enumerate(world.agents):
            if other is agent:
                aindex.append(ai)
                continue

            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + comm)

    def done(self, agent, world):
        occupied_landmarks = 0
        for l in world.landmarks:
            # 算出每个agent离地标的欧式距离
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.1:
                occupied_landmarks += 1
        return occupied_landmarks == len(world.landmarks)
