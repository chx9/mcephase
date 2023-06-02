import json
def read_config_file(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data