import numpy as np
from multiagent.core_prey_1 import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 12
        num_landmarks = 3
        num_obstacle = 0
        world.collaborative = True
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        world.obstacle = [Landmark() for i in range(num_obstacle)]
        for i, obstacle in enumerate(world.obstacle):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.1
            obstacle.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, obstacle in enumerate(world.obstacle):
            obstacle.color = np.array([0.55, 0.55, 0.55])
            obstacle.state.p_pos = np.array([0.0, 0.0])
            obstacle.state.p_vel = np.zeros(world.dim_p)

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
        p = []
        p.append(np.array([0.273221902, -0.4659316]))
        p.append(np.array([-0.54646218, -0.01075534]))
        p.append(np.array([0.25743576, 0.57392895]))
        p.append(np.array([0.5, -0.2]))
        p.append(np.array([-0.6, -0.6]))

        p.append(np.array([-0.06937189, 0.34963775]))
        p.append(np.array([0.54989629, 0.34437029]))
        p.append(np.array([0., -0.]))
        p.append(np.array([-0.14008428, - 0.36346792]))
        p.append(np.array([-0.12659007, 0.43926745]))
        p.append(np.array([0.72659007, 0.73926745]))
        p.append(np.array([-0.52659007, 0.33926745]))

        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_pos = p[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0

        for l in world.landmarks:
            cout_agent = 0
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            for d in dists:
                if d <= 0.1:
                    cout_agent += 1

            max = ((len(world.agents) / len(world.landmarks)) ** 2) * 5
            rew += (cout_agent ** 2) * 5 if cout_agent <= len(world.agents) / len(world.landmarks) else max
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame获取此代理的参考框架中所有实体的位置
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors

        aindex = []
        other_pos = []
        for ai, other in enumerate(world.agents):
            if other is agent:
                aindex.append(ai)
                continue
            # 观察其他agents和自己的距离
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

    def done(self, agent, world):
        occupied_landmarks = 0
        for l in world.landmarks:
            # 算出每个agent离地标的欧式距离
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.08:
                occupied_landmarks += 1
        return occupied_landmarks == 5
