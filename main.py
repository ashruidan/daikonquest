from utils import parse_arguments
from system import Environment

if __name__ == "__main__":
    args = parse_arguments()
    env = Environment(args)
    env.run()
    env.stop()
