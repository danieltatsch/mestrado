import json
import colorama
from termcolor import colored

def open_json_file(path : str) -> dict:
    data = {}

    try:
        with open(path) as json_file:
            data = json.load(json_file)
    except:
        pass

    return data

def debug(text, color=None, debug=True):
    if debug:
        print(colored(text, color)) if color is not None else print(text)