import os
import json
from trainer import main  

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    if not os.path.isdir(config["filepath"]):
        os.makedirs(config["filepath"])
    print("Configuration loaded:")
    print(json.dumps(config, indent=4))
    main(config)
