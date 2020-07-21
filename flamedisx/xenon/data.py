"""Tools to get data from other XENON repositories
"""
import getpass
import os
import os.path

import flamedisx as fd
export, __all__ = fd.exporter()


# Path to the root folder of the BBF repository, if you have already cloned BBF
# (Do not export it. If you do, reassigning fd.BBF_PATH won't update this one.)
BBF_PATH = './bbf'


@export
def pax_file(x):
    """Return URL to file hosted in the pax repository master branch"""
    return 'https://raw.githubusercontent.com/XENON1T/pax/master/pax/data/' + x


@export
def ensure_bbf(token=None):
    """Clones bbf (prompting for credentials) if we do not have it"""
    if not os.path.exists(BBF_PATH):
        print("XENON1T private data requested, we must clone BBF (~800 MB).")
        if token is None:
            print("    - Create a token with full 'repo' permissions at "
                  "https://github.com/settings/tokens\n"
                  "    - Save or write it down somewhere \n"
                  "    - Type it in the prompt above\n\n"
                  "'Repository not found' means you didn't give the token"
                  " full 'repo' permissions.\n"
                  "'Authentication failed' means you mistyped the token.\n")
            # We could prompt for username/password instead, but then the password
            # would be printed in plaintext if there is any problem during cloning.
            token = getpass.getpass('Github OAuth token: ')
        fd.run_command(f'git clone https://{token}:x-oauth-basic'
                       f'@github.com/XENON1T/bbf.git {BBF_PATH}')


@export
def get_bbf_file(data_file_name):
    """Return information from file in bbf/bbf/data/...

    Do NOT call on import time --
    that would make flamedisx unusable to non-XENON folks!
    """
    ensure_bbf()
    return fd.get_resource(f'{BBF_PATH}/bbf/data/{data_file_name}')
