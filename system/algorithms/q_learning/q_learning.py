import numpy as np

class Algorithm:
    def __init__(self, model, actions):
        self.Q = model
        self.history = []
        self.actions = actions

    def stop(self):
        return self.Q

    def train(self, data):
        state, reward, _, ratio,_,_ = data
        if len(self.history) == 0:
            return
        p_s,p_a = self.history[-1]
        lr = 0.8 - 0.7 * min(1,-2*ratio+2)
        g = 0.9
        print(self.Q)
        action = np.argmax(self.Q[state])
        self.Q[p_s][p_a] += lr * (r + g * action - self.Q[p_s][p_a])

    def step(self, data):
        state, _, actions, _, epsilon,_ = data
        if epsilon:
            action = actions[np.argmax(self.Q[state])]
        else:
            action = np.random.choice(actions, 1)[0]
        self.history.append((state, action))
        return action

    def last(self, data, done):
        checkpoint = data[5]
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
