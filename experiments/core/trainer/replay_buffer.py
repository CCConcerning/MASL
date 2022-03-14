import numpy as np
import random
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size,random_seed = 123):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        random.seed(random_seed)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes,Norm):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        #mean = np.mean(self._storage,axis=0)[2]
        #std = np.std(np.array(self._storage)[:,2],axis=0)
        #print('std:',std)
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))

            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        # if Norm:
        #     mean = np.mean(self._storage,axis=0)[2]
        #     std = np.std(np.array(self._storage)[:,2],axis=0)
        #     rew=np.nan_to_num((rewards-mean)/std)
        #     rewards = rew
            #print('rew',rew)
            #print('reward:',rewards)
        #print(rewards)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):

        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes,Norm=False):
        return self._encode_sample(idxes,Norm)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes,Norm=False)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer_goal(object):
    def __init__(self, size,random_seed = 123):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        random.seed(random_seed)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, goal,  done):
        data = (obs_t, action, reward, obs_tp1, goal,  done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes,Norm):
        obses_t, actions, rewards, obses_tp1,  goals, dones = [], [], [], [], [], []
        #mean = np.mean(self._storage,axis=0)[2]
        #std = np.std(np.array(self._storage)[:,2],axis=0)
        #print('std:',std)
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1,  goal, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))

            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            goals.append(np.array(goal, copy=False))
            #goals_tp1.append(np.array(goal_tp1, copy=False))
            dones.append(done)
        # if Norm:
        #     mean = np.mean(self._storage,axis=0)[2]
        #     std = np.std(np.array(self._storage)[:,2],axis=0)
        #     rew=np.nan_to_num((rewards-mean)/std)
        #     rewards = rew
            #print('rew',rew)
            #print('reward:',rewards)
        #print(rewards)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(goals), np.array(dones)

    def make_index(self, batch_size):

        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes,Norm=False):
        return self._encode_sample(idxes,Norm)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes,Norm=False)

    def collect(self):
        return self.sample(-1)

class ReplayBuffer_phase(object):
    def __init__(self, size,random_seed = 123):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        random.seed(random_seed)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, goal, phase, done):
        data = (obs_t, action, reward, obs_tp1, goal, phase, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes,Norm):
        obses_t, actions, rewards, obses_tp1,  goals, phases, dones = [], [], [], [], [], [], []
        #mean = np.mean(self._storage,axis=0)[2]
        #std = np.std(np.array(self._storage)[:,2],axis=0)
        #print('std:',std)
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1,  goal, phase, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))

            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            goals.append(np.array(goal, copy=False))
            phases.append(np.array(phase, copy=False))
            dones.append(done)
        # if Norm:
        #     mean = np.mean(self._storage,axis=0)[2]
        #     std = np.std(np.array(self._storage)[:,2],axis=0)
        #     rew=np.nan_to_num((rewards-mean)/std)
        #     rewards = rew
            #print('rew',rew)
            #print('reward:',rewards)
        #print(rewards)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(goals), np.array(phases), np.array(dones)

    def make_index(self, batch_size):

        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes,Norm=False):
        return self._encode_sample(idxes,Norm)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes,Norm=False)

    def collect(self):
        return self.sample(-1)

class ReplayBuffer_goal_policy(object):
    def __init__(self, size,random_seed = 123):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        random.seed(random_seed)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, accumulate_reward):
        data = (obs_t, accumulate_reward)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes,Norm):
        obses_t, rewards = [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, reward = data
            obses_t.append(np.array(obs_t, copy=False))
            rewards.append(reward)

        # if Norm:
        #     mean = np.mean(self._storage,axis=0)[2]
        #     std = np.std(np.array(self._storage)[:,2],axis=0)
        #     rew=np.nan_to_num((rewards-mean)/std)
        #     rewards = rew
            #print('rew',rew)
            #print('reward:',rewards)
        #print(rewards)
        return np.array(obses_t),  np.array(rewards)

    def make_index(self, batch_size):

        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes,Norm=False):
        return self._encode_sample(idxes,Norm)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes,Norm=False)

    def collect(self):
        return self.sample(-1)


class High_Value_ReplayBuffer(object):
    def __init__(self, size,random_seed = 123):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        random.seed(random_seed)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs):
        data = obs

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t = []

        for i in idxes:
            data = self._storage[i]
            obs_t = data
            obses_t.append(np.array(obs_t, copy=False))
        return np.array(obses_t)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha,random_seed):
        random.seed(random_seed)
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def make_index(self, batch_size):
        return self._sample_proportional(batch_size)
    def sample(self, batch_size, beta,Norm=False):
        #idxes = self._sample_proportional(batch_size)
        idxes = self.make_index(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes,Norm)
        return tuple(list(encoded_sample) + [weights,idxes])