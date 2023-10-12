import os
import json
import logging
from datetime import datetime
from omegaconf import OmegaConf

def init_logger(config):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    os.makedirs(os.path.join(config.run.outputs_dir, "logs"), exist_ok=True)
    log_path = os.path.join(config.run.outputs_dir, "logs", "{}.txt".format(current_time))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s || %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    
    logging.info("\n" + json.dumps(OmegaConf.to_container(config), indent=4, sort_keys=True))
