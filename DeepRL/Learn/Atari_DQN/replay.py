import numpy as np
import time
import random

from torch._C import Value

# This function was mostly pulled from
# https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=600000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32, num_heads=1, bernoulli_probability=1.0):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
            num_heads: integer number of heads needed in mask
            bernoulli_probability: bernoulli probability that an experience will go to a particular head
        """
        self.bernoulli_probability = bernoulli_probability
        assert(self.bernoulli_probability > 0)
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.count = 0
        self.current = 0
        self.num_heads = num_heads
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.masks = np.empty((self.size, self.num_heads), dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.float32)
        self.new_states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.float32)
        self.indices = np.empty(batch_size, dtype=np.int32)
        self.random_state = np.random.RandomState(393)
        if self.num_heads == 1:
            assert(self.bernoulli_probability == 1.0)

    def save_buffer(self, filepath):
        st = time.time()
        print("starting save of buffer to %s"%filepath, st)
        np.savez(filepath,
                 frames=self.frames, actions=self.actions, rewards=self.rewards,
                 terminal_flags=self.terminal_flags, masks=self.masks,
                 count=self.count, current=self.current,
                 agent_history_length=self.agent_history_length,
                 frame_height=self.frame_height, frame_width=self.frame_width,
                 num_heads=self.num_heads, bernoulli_probability=self.bernoulli_probability,
                 )
        print("finished saving buffer", time.time()-st)

    def load_buffer(self, filepath):
        st = time.time()
        print("starting load of buffer from %s"%filepath, st)
        npfile = np.load(filepath)
        self.frames = npfile['frames']
        self.actions = npfile['actions']
        self.rewards = npfile['rewards']
        self.terminal_flags = npfile['terminal_flags']
        self.masks = npfile['masks']
        self.count = npfile['count']
        self.current = npfile['current']
        self.agent_history_length = npfile['agent_history_length']
        self.frame_height = npfile['frame_height']
        self.frame_width = npfile['frame_width']
        self.num_heads = npfile['num_heads']
        self.bernoulli_probability = npfile['bernoulli_probability']
        if self.num_heads == 1:
            assert(self.bernoulli_probability == 1.0)
        print("finished loading buffer", time.time()-st)
        print("loaded buffer current is", self.current)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        mask = self.random_state.binomial(1, self.bernoulli_probability, self.num_heads)
        self.masks[self.current] = mask
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        return self.current-1

    def get_history(self, current_index):
        return self._get_state(current_index)

    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self, batch_size):
        if batch_size != self.indices.shape[0]:
             self.indices = np.empty(batch_size, dtype=np.int32)

        for i in range(batch_size):
            while True:
                index = self.random_state.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                # dont add if there was a terminal flag in previous
                # history_length steps
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self, batch_size):
        """
        Returns a minibatch of batch_size
        """
        if batch_size != self.states.shape[0]:
            self.states = np.empty((batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.float32)
            self.new_states = np.empty((batch_size, self.agent_history_length,
                                        self.frame_height, self.frame_width), dtype=np.float32)

        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices(batch_size)

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices], self.masks[self.indices]


    def __len__(self):
        return self.count


"""
https://github.com/labmlai/annotated_deep_learning_paper_implementations.git

---
title: Prioritized Experience Replay Buffer
summary: Annotated implementation of prioritized experience replay using a binary segment tree.
---

# Prioritized Experience Replay Buffer

This implements paper [Prioritized experience replay](https://papers.labml.ai/paper/1511.05952),
using a binary segment tree.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/dqn/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/fe1ad986237511ec86e8b763a2d3f710)
"""
class PriorityMemory:
    """
    ## Buffer for Prioritized Experience Replay

    [Prioritized experience replay](https://papers.labml.ai/paper/1511.05952)
     samples important transitions more frequently.
    The transitions are prioritized by the Temporal Difference error (td error), $\delta$.

    We sample transition $i$ with probability,
    $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$
    where $\alpha$ is a hyper-parameter that determines how much
    prioritization is used, with $\alpha = 0$ corresponding to uniform case.
    $p_i$ is the priority.

    We use proportional prioritization $p_i = |\delta_i| + \epsilon$ where
    $\delta_i$ is the temporal difference for transition $i$.

    We correct the bias introduced by prioritized replay using
     importance-sampling (IS) weights
    $$w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$$ in the loss function.
    This fully compensates when $\beta = 1$.
    We normalize weights by $\frac{1}{\max_i w_i}$ for stability.
    Unbiased nature is most important towards the convergence at end of training.
    Therefore we increase $\beta$ towards end of training.

    ### Binary Segment Tree
    We use a binary segment tree to efficiently calculate
    $\sum_k^i p_k^\alpha$, the cumulative probability,
    which is needed to sample.
    We also use a binary segment tree to find $\min p_i^\alpha$,
    which is needed for $\frac{1}{\max_i w_i}$.
    We can also use a min-heap for this.
    Binary Segment Tree lets us calculate these in $\mathcal{O}(\log n)$
    time, which is way more efficient that the naive $\mathcal{O}(n)$
    approach.

    This is how a binary segment tree works for sum;
    it is similar for minimum.
    Let $x_i$ be the list of $N$ values we want to represent.
    Let $b_{i,j}$ be the $j^{\mathop{th}}$ node of the $i^{\mathop{th}}$ row
     in the binary tree.
    That is two children of node $b_{i,j}$ are $b_{i+1,2j}$ and $b_{i+1,2j + 1}$.

    The leaf nodes on row $D = \left\lceil {1 + \log_2 N} \right\rceil$
     will have values of $x$.
    Every node keeps the sum of the two child nodes.
    That is, the root node keeps the sum of the entire array of values.
    The left and right children of the root node keep
     the sum of the first half of the array and
     the sum of the second half of the array, respectively.
    And so on...

    $$b_{i,j} = \sum_{k = (j -1) * 2^{D - i} + 1}^{j * 2^{D - i}} x_k$$

    Number of nodes in row $i$,
    $$N_i = \left\lceil{\frac{N}{D - i + 1}} \right\rceil$$
    This is equal to the sum of nodes in all rows above $i$.
    So we can use a single array $a$ to store the tree, where,
    $$b_{i,j} \rightarrow a_{N_i + j}$$

    Then child nodes of $a_i$ are $a_{2i}$ and $a_{2i + 1}$.
    That is,
    $$a_i = a_{2i} + a_{2i + 1}$$

    This way of maintaining binary trees is very easy to program.
    *Note that we are indexing starting from 1*.

    We use the same structure to compute the minimum.
    """

    def __init__(self, alpha=0.6, size=650000, state_shape=(650000, 84, 84)):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = size
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'frame': np.zeros(shape=state_shape, dtype=np.float32),
            'action': np.zeros(shape=size, dtype=np.int32),
            'reward': np.zeros(shape=size, dtype=np.float32),
            'done': np.zeros(shape=size, dtype=np.bool)
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add_experience(self, action, frame, reward, done):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['frame'][idx] = frame
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

        return idx

    def get_history(self, current_index):
        c3 = current_index
        c2 = self._get_valid_index(current_index-1) 
        c1 = self._get_valid_index(current_index-2) 
        c0 = self._get_valid_index(current_index-3) 
        frames = np.array(
            [self.data['frame'][c0], self.data['frame'][c1], self.data['frame'][c2], self.data['frame'][c3]])
        return frames

    def get_frame(self, current_index):
        return np.array([
                            self.data['frame'][current_index]
                        ])

    def get_minibatch(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        # https://github.com/younggyoseo/Ape-X/blob/master/memory.py modification following this. 
        p_total = self._sum()

        # samples from the equal probability range
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._find_prefix_sum_idx(mass)
            # make within size range. 
            samples['indexes'][i] = idx % self.size

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            assert prob > 0
            assert self.size > 0
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        # Build states, actions, rewards, next_states, dones, weights, indices np arrays  
        states = []
        next_states = []
        actions = []
        dones = []
        rewards = []
        for i in range(batch_size):
            states.append(self.get_history(samples['indexes'][i] ))
            actions.append(self.data['action'][samples['indexes'][i]])
            rewards.append(self.data['reward'][samples['indexes'][i]])
            next_states.append(self.get_history(self._get_next_index(samples['indexes'][i]) ))
            dones.append(self.data['done'][samples['indexes'][i]])
            
        return  (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones), 
            samples['weights'], 
            samples['indexes']
        ) 

    def get_minibatch_normal(self, batch_size):
        """
        ### Sample from buffer without using priority
        """

        indices = np.random.choice(self.size-1, batch_size)

        # Build states, actions, rewards, next_states, dones, weights, indices np arrays  
        states = []
        next_states = []
        actions = []
        dones = []
        rewards = []
        for i in range(batch_size):
            states.append(self.get_history(indices[i]))
            actions.append(self.data['action'][indices[i]])
            rewards.append(self.data['reward'][indices[i]])
            next_states.append(self.get_history(self._get_next_index(indices[i])))
            dones.append(self.data['done'][indices[i]])
            
        return  (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones), 
        ) 

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            assert priority > 0
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)
            # priority is a just td-error. it becomes prob only after divided by _sum()
            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def _find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size

    def get_size(self):
        ''' 
        Checks the initial size of buffer slots used. 
        '''
        return self.size

    def _get_valid_index(self, index):
        if index >= self.capacity:
            raise ValueError(f'index {index} is out of {self.capacity} bound')
        if index < 0 : 
            index = self.capacity + index
        return index
        
    def _get_next_index(self, index):
        if index+1 >= self.capacity:
            return 0
        if index < 0 : 
            raise ValueError(f'index {index} is negative')
        return index+1

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """
        # The root node keeps the minimum of all values
        return self.priority_min[1]