import numpy as np
from system import Actions
from games.pokemon_red import local_to_global

class Custom:
    def __init__(self):
        E = 3
        Y = 90
        X = 124
        self.model_size = E * Y * X
        self.batch_e = 0
        self.episode_e = 0
        self.eps = 1
        self.battle_flag = None
        self.start = None
        self.checkpoint = None
        self.previous = None
        self.continuous = 0
        self.overworld = False

        # To run the modified actions, the existing model must be deleted
        actions = Actions.list()
        actions.remove(Actions.B.value)
        actions.remove(Actions.START.value)
        actions.remove(Actions.SELECT.value)
        self.actions = actions

    def custom(self, train, memory, checkpoint, action):
        self.memory = memory
        state, most, done = self.state()
        reward = self.reward(state, most)
        action = self.battle(memory[0xD057], memory[0xCC25],memory[0xCC26])
        lr = self.lr()
        epsilon = self.epsilon(train)
        return (action, state, reward, lr, epsilon, most, done)

    def state(self):
        done = False
        most = False
        e = self.memory[0xD74E]
        e = e // 2 + e % 2
        self.episode_e = e
        y = self.memory[0xD361]
        x = self.memory[0xD362]
        map = self.memory[0xD35E]
        y, x = local_to_global(y,x,map)
        y -= 267
        x -= 36
        if (x == 55 or x == 56 or x == 57) and y == 7:
            done = True
        if self.start == None or self.checkpoint[0] != e:
            self.start = (e,y,x)
            self.checkpoint = (e,y,x)
            self.overworld = map < 15
            most = True
        else:
            _,sy,sx = self.start
            _,cy,cx = self.checkpoint
            if self.overworld == map < 15 and abs((y+x) - (sy+sx)) > abs((cy+cx) - (sy+sx)):
                self.checkpoint = (e,y,x)
                self.overworld = map < 15
                most = True
            elif not self.overworld:
                self.checkpoint = (e,y,x)
                self.overworld = map < 15
                most = True
        return (e * (89 * 123) + y * 123 + x, most, done)

    def reward(self, state, most):
        if self.battle_flag != None:
            return 0
        reward = 2
        if self.previous == None:
            self.previous = state
        if self.previous == state:
            reward /= -(5+self.continuous)
            self.continuous += 1
        if most:
            reward *= 10
            self.continuous = 0
            self.continuous = 0
        return reward

    def lr(self):
        if self.episode_e > self.batch_e:
            self.batch_e = self.episode_e
        ratio = (self.episode_e + 1) / (self.batch_e + 1)
        return 0.8 - 0.7 * min(1,-2*ratio+2)

    def epsilon(self, train):
        if not train:
            return True
        self.eps = max(0.1,self.eps*0.994)
        return self.eps < np.random.random()

    def battle(self, type, x, select):
        if not type:
            self.battle_flag = None
            return None
        if type == 2:
            return Actions.A.value
        if type == 1:
            if self.battle_flag == None:
                p_x, p_s, a = (0,0,'')
            else:
                p_x,p_s,a = self.battle_flag
            match (x, select):
                case (15,1):
                    action = Actions.A.value
                case (9,1):
                    action = Actions.RIGHT.value
                case (p_x, p_s):
                    if a == Actions.DOWN.value:
                        action = Actions.A.value
                    else:
                        action = Actions.DOWN.value
                case _:
                    action = Actions.DOWN.value
            self.battle_flag = (x, select, action)
            return action
