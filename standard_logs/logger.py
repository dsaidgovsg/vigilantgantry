# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
The setup_logging function provide customs stdout for debug and info. 

Author: GovTech
"""
import yaml
import logging
import logging.config


def setup_logging(
    default_path="standard_logs/logging.yaml", default_level=logging.DEBUG
):
    """
    Setup logging

    :param default_path: Path to yaml file, defaults to "standard_logs/logging.yaml"
    :type default_path: str, optional
    :param default_level: Logger default file, defaults to logging.DEBUG
    :type default_level: logging class, optional
    """
    with open(default_path, "rt") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
