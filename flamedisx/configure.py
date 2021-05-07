import flamedisx as fd

import configparser, os
import importlib

def configure_detector(detector='default'):
    fd.detector = detector

    assert os.path.exists(os.path.join(
    os.path.dirname(__file__), 'config/', detector+'.ini'))
    fd.config = configparser.ConfigParser(inline_comment_prefixes=';')
    fd.config.read(os.path.join(os.path.dirname(__file__), 'config/', detector+'.ini'))
