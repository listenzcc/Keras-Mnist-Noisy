'''
FileName: __init__.py
Author: Chuncheng
Version: V0.0
Purpose: Package Init
'''

import os
import logging
import configparser


def mk_logger(name, level, fmt):
    '''
    Method:mk_logger

    Make Logger Object as the name, level and fmt.

    Args:
    - @name: The name of the logger;
    - @level: The logging level;
    - @fmt: The fmt of the logger entry.

    Outputs:
    - The logger object.

    '''
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ------------------------------------------------------------------------------
# Config
pkg_folder = os.path.dirname(__file__)
Cfg = configparser.ConfigParser()
Cfg.read(os.path.join(pkg_folder, 'setting.cfg'))


# ------------------------------------------------------------------------------
# Logger
kwargs = dict(
    name=Cfg['Default']['pkgName'],
    level=logging.DEBUG,
    fmt='%(asctime)s - %(levelname)s - %(message)s - (%(filename)s %(lineno)d)'
)
logger = mk_logger(**kwargs)
logger.info('Package Initialized')

# ------------------------------------------------------------------------------
# Paths
noisyFolder = Cfg['Default']['noisyFolder']
