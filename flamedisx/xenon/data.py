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

# Path to the root folder of the XENONnT/Flamedisx repository, if you have
# already cloned XENONnT/Flamedisx
# (Do not export it. If you do, reassigning fd.NTFD_PATH won't update this one.)
NTFD_PATH = './flamedisx_maps'


@export
def pax_file(x):
    """Return URL to file hosted in the pax repository master branch"""
    return 'https://raw.githubusercontent.com/XENON1T/pax/master/pax/data/' + x


@export
def ensure_token(token=None):
    """Requests for token if token is not already available."""
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
    return token


@export
def ensure_repo(repo_name, repo_path, token=None):
    """Clones private repository (prompting for credentials) if we do not have it"""
    if not os.path.exists(repo_path):
        print("Private data requested, we must clone repository folder.")
        token = ensure_token()
        fd.run_command(f'git clone https://{token}:x-oauth-basic'
                       f'@github.com/{repo_name} {repo_path}')


@export
def get_bbf_file(data_file_name):
    """Return information from file in bbf/bbf/data/...

    Do NOT call on import time --
    that would make flamedisx unusable to non-XENON folks!
    """
    ensure_repo('XENON1T/bbf.git', BBF_PATH)
    return fd.get_resource(f'{BBF_PATH}/bbf/data/{data_file_name}')


@export
def get_nt_file(data_file_name):
    """Return information from file in XENONnT/Flamedisx/...

    Do NOT call on import time --
    that would make flamedisx unusable to non-XENON folks!
    """
    ensure_repo('XENONnT/Flamedisx.git', NTFD_PATH)
    return fd.get_resource(f'{NTFD_PATH}/{data_file_name}')


@export
def refresh_nt(token=None):
    """Cloning latest version of XENONnT/Flamedisx (prompting for credentials) if we do not have it"""
    # `git pull` does not work if not in a git repo and not everybody wants to `git init`
    # Delete old repo and clone again
    if os.path.exists(NTFD_PATH):
        fd.run_command(f'rm -rf {NTFD_PATH}')
    print("XENONnT private data requested, we are cloning latest Flamedisx folder.")
    token = ensure_token()
    fd.run_command(f'git clone https://{token}:x-oauth-basic'
                   f'@github.com/XENONnT/Flamedisx.git {NTFD_PATH}')
