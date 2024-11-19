import argparse, os

def argument_list(path):
    return [
        f for f in os.listdir(path) 
        if os.path.isdir(os.path.join(path, f))
    ]

def parse_arguments(games, agents):
    parser = argparse.ArgumentParser(
        description="Run an Agent to play any GameBoy game from a selected list"
    )
    parser.add_argument(
        "-game", "--game", help="Select the game to play",
        choices=argument_list(games), required=True
    )
    parser.add_argument(
        "-agent", "--agent",
        help="Select the agent with specific algorithm",
        choices=argument_list(agents), required=True
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-headless","--headless", action="store_true", help="Run Headless Parallel Training mode"
    )
    mode.add_argument(
        "-human","--human", action="store_true", help="Run Human Control mode"
    )
    mode.add_argument(
        "-train","--training", action="store_true", help="Run Graphical Training mode"
    )
    mode.add_argument(
        "-eval","--evaluation", action="store_true", help="Run Evaluation mode"
    )
    return parser.parse_args()
