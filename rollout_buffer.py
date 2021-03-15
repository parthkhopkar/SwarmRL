import random
from collections import deque
import numpy as np


class NStepRolloutBuffer:
    def __init__(self, n, capacity=5000, gamma=0.95, num_states=1):
        self.n = n  # Max number of rollout per trajectory
        self.capacity = int(capacity)
        self.gamma = gamma

        self._num_states = num_states

        self.state_buffer = []
        self.action_buffer = []
        self.log_prob_buffer = []
        self.reward_buffer = []
        self.mask_buffer = []

        self.memory = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)

    def path_end(self):
        return len(self.reward_buffer) >= self.n

    def add_transition(self, state, action, reward, log_prob, mask):
        if self.path_end():
            raise PathEndError(f'Failed to add transition beyond {self.n} steps')

        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.log_prob_buffer.append(log_prob)
        self.reward_buffer.append(reward)
        self.mask_buffer.append(mask)

    def finish_path(self, next_value):
        v = next_value
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            # print('REWARD', r)
            # print('VALUE', v)
            v = r + self.gamma * v
            # print('DISCOUNTED REWARD', v)
            discounted_r.append(v)

        discounted_r = discounted_r[::-1]
        # print('DISCOUNTED REWARD BUFFER SHAPE', np.array(discounted_r).shape)

        for i in range(len(self.action_buffer)):
            memory_batch = zip([[self.state_buffer[i][0][j], self.state_buffer[i][1][j]] for j in range(len(self.state_buffer[i][0]))], 
                                 self.action_buffer[i].numpy(), 
                                 discounted_r[i],
                                 self.log_prob_buffer[i].numpy(), 
                                 self.mask_buffer[i])
            self.memory.extend(memory_batch)

        self.clear_cache()

    def get_buffer(self, batch_size):
        if batch_size > len(self.memory):
            batch = self.memory
        else:
            batch = random.sample(self.memory, batch_size)

        return self._to_numpy(batch)

    def _to_numpy(self, experiences):
        states, actions, rewards_to_go, log_probs, masks = zip(*experiences)

        if self._num_states > 1:
            states = [np.array(state, dtype=np.float32) for state in zip(*states)]
        else:
            states = np.array(states, dtype=np.float32)

        actions = np.array(actions, dtype=np.float32)
        rewards_to_go = np.expand_dims(rewards_to_go, -1).astype(np.float32)
        log_probs = np.array(log_probs, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        return states, actions, rewards_to_go, log_probs, masks

    def clear_cache(self):
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.log_prob_buffer.clear()
        self.reward_buffer.clear()
        self.mask_buffer.clear()


class PathEndError(Exception):
    def __init__(self, message):
        super().__init__(message)
