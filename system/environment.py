import os, importlib, numpy as np
from pyboy import PyBoy
from utils import load, load_pickle, save_pickle
from .agent import Agent

class Environment:
    def __init__(self, args):
        self.headless = args.headless
        self.human = args.human
        self.train = args.training
        self.eval = args.evaluation

        game = args.game
        algorithm = args.algorithm

        # ROM PATH
        rom = f"games/{game}/{game}.gb"
        # START_SAVE PATH
        start_save = f"games/{game}/start.save"
        self.start_save = start_save
        # ALGORITHM_IMPORT PATH
        algorithm_import = f"system.algorithms.{algorithm}"
        # MODEL PATH
        self.model = f"games/{game}/{algorithm}.model"
        # CUSTOM_IMPORT PATH
        custom_import = f"games.{game}.{algorithm}"

        self.init_emulator(rom, start_save, args.headless)
        self.init_game(custom_import)
        self.init_agent(algorithm_import, self.model)

    def init_emulator(self, rom, start_save, headless):
        win, spd = ("null", 0) if headless else ("SDL2", 5)
        self.emulator = PyBoy(rom, window=win)
        if not self.emulator:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM")
        self.emulator.set_emulation_speed(spd)
        load(start_save, "rb", self.emulator.load_state)

    def init_game(self, custom):
        m = importlib.import_module(custom)
        self.custom = getattr(m,"Custom")()

    def init_agent(self, algo_import, model_path):
        model = None
        if os.path.exists(model_path):
            model = load_pickle(model_path)
        else:
            model = np.zeros((self.custom.model_size(), len(self.custom.actions())))
        self.agent = Agent(algo_import, model, self.custom.actions())

    def stop(self):
        save_pickle(self.model, self.agent.stop())
        self.emulator.stop(False)

    def run(self):
        try:
            self.batch()
        except KeyboardInterrupt:
            print("Program interrupted. Stopping emulator...")
        finally:
            self.stop()

    def controller_input(self, input):
        self.emulator.button_press(input)
        self.emulator.tick(20)
        self.emulator.button_release(input)
        self.emulator.tick(2)

    def batch(self):
        batch_size = 100
        episode = 0
        while episode < batch_size:
            self.episode()
            load(self.start_save, "rb", self.emulator.load_state)
            episode += 1

    def episode(self):
        episode_size = 25000
        step = 0
        actions = self.custom.actions()
        done = False
        a = None
        while not done and step < episode_size:
            state = self.custom.state(self.emulator.memory)
            custom = self.custom.custom(a)
            reward,done = self.custom.reward(state, a)
            algorithm = self.custom.algorithm(step, self.eval)
            data = (state, reward, actions) + algorithm #(ratio, epsilon, checkpoint)
            if not self.human:
                if self.train or self.headless:
                    self.agent.train(data)
                a = self.agent.step(data,custom)
                self.controller_input(a)
            else:
                self.agent.debug(data)
            self.emulator.tick()
            step += 1
        self.agent.last(data, done)
