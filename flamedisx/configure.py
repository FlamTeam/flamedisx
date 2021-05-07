import flamedisx as fd

def configure_mode(mode = 'default'):
    if mode == 'default':
        fd.mode_folder = mode
        configure_detector('default')
    else:
        raise NotImplementedError

def configure_detector(detector):
    if detector == 'default':
        assert fd.mode_folder in ('default',)
        fd.config_file = detector + '_config.ini'
    elif detector == 'LUX':
        assert fd.mode_folder in ('nest',)
        fd.config_file = detector + '_config.ini'
    else:
        raise NotImplementedError
