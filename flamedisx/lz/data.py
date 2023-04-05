"""Tools to get private LZ data
"""
import getpass
import os
import os.path

import flamedisx as fd
export, __all__ = fd.exporter()


# Path to the root folder of the private LZ data repository, if you have already cloned it
PATH = './lz_private_data'


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


def ensure_repo(repo_name, repo_path, token=None):
    """Clones private repository (prompting for credentials) if we do not have it"""
    if not os.path.exists(repo_path):
        print("Private data requested, we must clone repository folder.")
        token = ensure_token()
        fd.run_command(f'git clone https://{token}:x-oauth-basic'
                       f'@github.com/{repo_name}')


@export
def get_lz_file(data_file_name):
    """Return information from file in lz_private_data/...
    """
    ensure_repo('robertsjames/lz_private_data.git', PATH)
    return fd.get_resource(f'{PATH}/{data_file_name}')
