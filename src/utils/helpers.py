# src/utils/helpers.py

import os
import json
from datetime import datetime


def save_json(data: dict, file_path: str):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def timestamp():

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")