import argparse

parser = argparse.ArgumentParser(description='take the config path.')
parser.add_argument('config_path', metavar='N', type=str)


args = parser.parse_args()

if __name__ == '__main__':
    print(args.config_path)