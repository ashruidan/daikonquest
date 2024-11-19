import os, importlib, numpy as np
from pyboy import PyBoy
from utils import load, load_pickle, save_pickle
from system import Actions

class Environment:
    def __init__(self, agent, game, headless):
        # PATH
        custom_path = f"games.{game}"
        rom_path = f"games/{game}/{game}.gb"
        agent_path = f"agents.{agent}"
        self.start_save_path = f"games/{game}/start.save"
        self.model_path = f"games/{game}/model.pkl"

        # VARIABLE
        self.batch = 60*7
        self.episode = 25000
        if headless:
            window = "null"
            speed = 0
        else:
            window = "SDL2"
            speed = 5

        # GAME
        custom_import = importlib.import_module(custom_path)
        self.custom = getattr(custom_import, "Custom")()
        self.actions = self.custom.actions

        # EMULATOR
        self.emulator = PyBoy(rom_path, window=window)
        if not self.emulator:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM")
        self.emulator.set_emulation_speed(speed)

        # AGENT
        if os.path.exists(self.model_path):
            model = load_pickle(self.model_path)
        else:
            model = np.ones((self.custom.model_size, len(self.actions)))
        agent_import = importlib.import_module(agent_path)
        self.agent = getattr(agent_import,"Agent")(model, self.actions)

    def stop(self):
        save = self.agent.stop()
        save_pickle(self.model_path, save)
        self.emulator.stop()

    def run(self, train, human):
        try:
            # Batch
            episode = 1
            while episode <= self.batch:
                # Episode
                load(self.start_save_path, "rb", self.emulator.load_state)
                step = 0
                done = False
                checkpoint = None
                action = None
                state = None
                while step < self.episode and not done:
                    if step % 5000 == 0:
                        print(f"Episode: {episode}, Step: {step}")
                    data = self.custom.custom(train, self.emulator.memory, checkpoint, action)
                    action, state, reward, lr, epsilon, most, done = data
                    if not human:
                        if action == None:
                            if train:
                                self.agent.train(state, reward, lr)
                            action = self.agent.step(state, epsilon)
                        self.emulator.button_press(action)
                        self.emulator.tick(20)
                        self.emulator.button_release(action)
                        self.emulator.tick(2)
                        if most:
                            checkpoint = (state, action)
                    else:
                        Q = self.agent.Q
                        if state != None:
                            print(f"{state}, {Q[state]}")
                    self.emulator.tick()
                    step += 1
                if not human:
                    self.agent.last(checkpoint, done)
                episode += 1
        except KeyboardInterrupt:
            print("Program interrupted. Stopping emulator...")
        finally:
            self.stop()
