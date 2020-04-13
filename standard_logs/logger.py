import yaml
import logging
import logging.config

def setup_logging(default_path='standard_logs/logging.yaml', default_level=logging.DEBUG):
    with open(default_path, 'rt') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)