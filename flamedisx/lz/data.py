"""Tools to get private LZ data
"""
import os
import os.path
import sys

import flamedisx as fd
export, __all__ = fd.exporter()


@export
def lz_setup(mode='tensors', run='sr1', lz_private_data_dir=None):
    """Update the system path to include necessary LZ private data files
    """
    assert lz_private_data_dir is not None, 'Must specifcy a private data directory!'

    if os.path.isdir(f'{lz_private_data_dir}/{mode}/{run}'):
        sys.path.append(f'{lz_private_data_dir}/{mode}/{run}')
        print(f'Successfully found mode {mode} and run {run} in ' + \
                f'directory {lz_private_data_dir}')
    else:
        raise RuntimeError(f'Error finding mode {mode} and run {run} in directory {lz_private_data_dir}')
