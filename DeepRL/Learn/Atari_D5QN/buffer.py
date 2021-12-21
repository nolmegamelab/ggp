# code from openai
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import random
import operator
import numpy as np

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

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
        indices = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(indices)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
        PrioritizedReplayBuffer merged with CustomizedPrioritizedReplayBuffer class
        1. Edited add method to receive priority as input. This enables to enter priority when adding sample.
           This efficiently merges two methods (add, update_priorities) which enables less shared memory lock.
        2. If we save obs as numpy.array, this will decompress LazyFrame which leads to memory explosion.
           To achieve memory efficiency, It is necessary to remove np.array(obs) from _encode_sample.
    """

    def __init__(self, capacity, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done, priority):

        if priority < 0 : priority = 0

        idx = self._next_idx
        data = (state, action, reward, next_state, done, priority)

        if len(self._storage) < self._maxsize:
            if self._next_idx < len(self._storage):
                self._storage[self._next_idx] = data
            else:
                self._storage.append(data)
        else:
            self._storage[self._next_idx] = data # overwrite

        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta, wire=False):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and indices
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
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
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        indices: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        indices = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        p_min = max(p_min, 0.0001)
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        if not wire:
            weights = np.array(weights)
        encoded_sample = self._encode_sample(indices, wire)

        return tuple(list(encoded_sample) + [weights, indices])

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index indices[i] in buffer
        to priorities[i].
        Parameters
        ----------
        indices: [int]
            List of indices of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled indices denoted by
            variable `indices`.
        """
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def _encode_sample(self, indices, wire=False):
        states, actions, rewards, next_states, dones, priorities = [], [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done, priority = data
            if wire:
                states.append(state.tolist())
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state.tolist())
                dones.append(done)
                priorities.append(priority)
            else:
                states.append(state)
                actions.append(np.array(action, copy=False))
                rewards.append(np.array(reward, copy=False))
                next_states.append(next_state)
                dones.append(np.array(done, copy=False))
                priorities.append(priority)
        # end for indices

        if wire:
            return (states,
                    actions,
                    rewards,
                    next_states,
                    dones, 
                    priorities)
        else:
            return (states,
                    np.array(actions),
                    np.array(rewards),
                    next_states,
                    np.array(dones), 
                    np.array(priorities))
    # end _encode_sample

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            # PER paper : uniform sampling over priority range 
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res