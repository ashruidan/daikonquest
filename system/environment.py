import os, importlib, numpy as np
from pyboy import PyBoy
from utils import load, load_pickle
from .actions import Actions
from games.pokemon_red.global_map import local_to_global

class Environment:
    def __init__(self, algorithm, game, headless):
        # PATH
        custom_path = f"games.{game}.{algorithm}"
        rom_path = f"games/{game}/{game}.gb"
        agent_path = f"system.algorithms.{algorithm}"
        self.start_save_path = f"games/{game}/start.save"
        model_path = f"games/{game}/{algorithm}.model"

        # VARIABLE
        if headless:
            window = "null"
            speed = 0
        else:
            window = "SDL2"
            speed = 5
        if os.path.exists(model_path):
            model = load_pickle(model_path)
        else:
            model = np.zeros((self.custom.model_size(), len(Actions.list())))

        # GAME
        custom_import = importlib.import_module(custom_path)
        self.custom = getattr(custom_import, "Custom")()

        # EMULATOR
        self.emulator = PyBoy(rom_path, window=window)
        if not self.emulator:
            raise RuntimeError("Failed to initialize PyBoy with the given ROM")
        self.emulator.set_emulation_speed(speed)

        # AGENT
        agent_import = importlib.import_module(agent_path)
        self.agent = getattr(agent_import,"Algorithm")(model, Actions.list())

    def stop(self):
        return 0

    def run(self, train, human):
        try:
            # Batch
            while True:
                # Episode
                load(self.start_save_path, "rb", self.emulator.load_state)
                while True:
                    action = self.custom.custom()
                    if action == None:
                        e = self.emulator.memory[0xD74E]
                        e = e // 2 + e % 2
                        y, x = local_to_global(self.emulator.memory[0xD361],self.emulator.memory[0xD362],self.emulator.memory[0xD35E])
                        y -= 267
                        x -= 36
                        action = self.agent.step(e * (89 * 123) + y * 123 + x,0.75 > np.random.random())
                    self.emulator.button_press(action)
                    self.emulator.tick(20)
                    self.emulator.button_release(action)
                    self.emulator.tick(2)
                    self.emulator.tick()
        except KeyboardInterrupt:
            print("Program interrupted. Stopping emulator...")
        finally:
            self.stop()

#         lr = 0.8 - 0.7 * min(1,-2*ratio+2)
