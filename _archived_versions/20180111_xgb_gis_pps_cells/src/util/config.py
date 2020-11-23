import json
import os


def load_settings(config_path='../settings.user.json', root_dir=None):
    if root_dir is None:
        root_dir = os.path.dirname(__file__)

    path = os.path.join(root_dir, config_path)
    with open(path, mode='r') as file:
        return json.load(file)
