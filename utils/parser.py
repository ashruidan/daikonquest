import argparse, os

def argument_list(path):
    return [
        f for f in os.listdir(path) 
        if os.path.isdir(os.path.join(path, f))
    ]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run an AI to play any GameBoy game from a selected list"
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

    # GAME LIST PATH
    parser.add_argument(
        "-game", "--game", help="Select the game to play",
        choices=argument_list("./games"), required=True
    )

    # ALGORITHM LIST PATH
    parser.add_argument(
        "-algo", "--algorithm", 
        help="Select the algorithm that will be used by agent",
        choices=argument_list("./system/algorithms"), required=True
    )
    return parser.parse_args()
