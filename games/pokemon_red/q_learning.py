import numpy as np
from system import Actions
from .global_map import local_to_global

class Custom:
    def __init__(self):
        self.previous = None
        self.current = None
        self.checkpoint = None
        self.distance = 0
        self.breadcrumb = None
        self.last = None
        self.battle = None
        self.pbattle = None
        self.batch = 0

    def reset(self):
        self.previous = None
        self.current = None
        self.checkpoint = None
        self.distance = 0
        self.breadcrumb = None
        self.last = None
        self.battle = None
        self.pbattle = None

    def model_size(self):
        E = 3
        Y = 90
        X = 124
        return E*Y*X

    def state(self, mem):
        event = mem[0xD74E]
        pos_y = mem[0xD361]
        pos_x = mem[0xD362]
        pos_map = mem[0xD35E]
        battle_type = mem[0xD057]
        battle_x = mem[0xCC25]
        battle_select = mem[0xCC26]

        self.pbattle = self.battle
        self.battle = (battle_type, battle_x, battle_select)

        e = event // 2 + event % 2
        y,x = local_to_global(pos_y,pos_x,pos_map)
        y -= 267
        x -= 36
        self.current = (e, y, x)
        return e * (89 * 123) + y * 123 + x

    def reward(self, state):
        reward = -1
        done = False
        e,y,x = self.current
        if self.previous == None:
            self.checkpoint = (y,x)
            self.breadcrumb = (y,x)
            self.last = state
            return (reward, done)
        if self.previous != self.current:
            self.checkpoint = (y,x)
            self.breadcrumb = (y,x)
            self.last = state
            reward += 1000
        pe,py,px,pa = self.previous
        if (e,y,x) == (pe,py,px) and pa != 'a':
            reward -= 10
        current_distance = abs(self.checkpoint[0] - self.breadcrumb[0]) + abs(self.checkpoint[1] - self.breadcrumb[1])
        if current_distance > self.distance:
            self.distance = current_distance
            self.breadcrumb = (y,x)
            self.last = state
        if (x == 55 or x == 56 or x == 57) and y == 7:
            done = True
        return (reward,done)

    def actions(self):
        actions = Actions.list()
        actions.remove(Actions.B.value)
        actions.remove(Actions.START.value)
        actions.remove(Actions.SELECT.value)
        return actions

    def algorithm(self, step):
        if self.batch < self.current[0]:
            self.batch = self.current[0]
        ratio = (self.current[0] + 1) / (self.batch + 1)
        epsilon = max(0.1, (self.distance + 1) / (step + 1)) > np.random.random()
        return (ratio, epsilon, self.last)

    def custom(self, a):
        type, x, select = self.battle
        if not type:
            return None
        if type == 2:
            return Actions.A.value
        if type == 1:
            flag = self.battle == self.pbattle and a == Actions.DOWN.value
            if (x,select) == (9,1):
                action = Actions.RIGHT.value
            elif (x,select) == (15,1) or flag:
                action = Actions.A.value
            else:
                action = Actions.DOWN.value
        return action

