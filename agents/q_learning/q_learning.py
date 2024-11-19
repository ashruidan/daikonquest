import numpy as np

class Agent:
    def __init__(self, model, actions):
        self.Q = model
        self.history = []
        self.actions = actions

    def stop(self):
        return self.Q

    def train(self, state, reward, lr):
        if len(self.history) == 0:
            return
        s,a = self.history[-1]
        g = 0.9
        action = np.argmax(self.Q[state])
        a = self.actions.index(a)
        self.Q[s][a] += lr * (reward + g * action - self.Q[s][a])

    def step(self, state, epsilon):
        if epsilon:
            i = np.random.choice(np.flatnonzero(np.isclose(self.Q[state],np.max(self.Q[state]))))
            action = self.actions[i]
        else:
            action = np.random.choice(self.actions, 1)[0]
        self.history.append((state, action))
        return action

    def last(self, checkpoint, done):
        i = len(self.history) - 1
        for index, (s,a) in enumerate(reversed(self.history)):
            if s == checkpoint:
                i = len(self.history) - index
        history = self.history[:i]
        seen = set()
        optimal = []
        for s, a in reversed(history):
            if s not in seen:
                seen.add(s)
                optimal.append((s,a))
            else:
                while optimal[-1][0] != s:
                    seen.discard(optimal[-1][0])
                    optimal.pop()
        value = 0
        for it in range(0,len(optimal)):
            state, action = optimal[it]
            if it == 0:
                value = np.argmax(self.Q[state]) + done * 10000
            else:
                a = self.actions.index(action)
                self.Q[state][a] += 0.1 * value
                value = self.Q[state][a]
        self.history.clear()
