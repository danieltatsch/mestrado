from utils import *

def main():
    config_path = "config.json"

    config = open_json_file(config_path)
    print(config)

if __name__ == "__main__":
    main()