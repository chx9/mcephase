import json
from utils import read_config_file
pth = 'data/frame_info.json'
info = read_config_file(pth)
import json
from sklearn.model_selection import train_test_split

# Assuming you have the dictionary 'info'

# Get the keys from the 'info' dictionary
keys = list(info.keys())

# Split the keys into train and test sets
train_keys, test_keys = train_test_split(keys, test_size=0.3, random_state=42)

# Create the train and test dictionaries
train_info = {key: info[key] for key in train_keys}
test_info = {key: info[key] for key in test_keys}

# Specify the file paths for train and test JSON files
train_file_path = 'data/train_info.json'
test_file_path = 'data/test_info.json'

# Save the train dictionary to a JSON file
with open(train_file_path, 'w') as train_file:
    json.dump(train_info, train_file)

# Save the test dictionary to a JSON file
with open(test_file_path, 'w') as test_file:
    json.dump(test_info, test_file)
