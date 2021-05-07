import flamedisx as fd
import os

def configure_detector(detector='default'):
    fd.detector = detector
    fd.config_file = detector + '_config.ini'
    assert os.path.exists(os.path.join(
    os.path.dirname(__file__), 'config', fd.config_file))
