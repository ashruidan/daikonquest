from utils import parse_arguments
from system import Environment

if __name__ == "__main__":
    args = parse_arguments()
    env = Environment(args.algorithm, args.game, args.headless)
    env.run(args.headless or args.training, args.human)
