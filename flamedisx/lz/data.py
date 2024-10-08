"""Tools to get private LZ data
"""
import getpass
import os
import os.path
import random
import string

import flamedisx as fd
export, __all__ = fd.exporter()


# Path to the root folder of the private LZ data repository, if you have already cloned it
PATH = 'lz_private_data'


def ensure_token(token=None):
    """Requests for token if token is not already available."""
    if token is None:
        print("    - Create a token with 'read repository' permissions at "
              "https://gitlab.com/-/profile/personal_access_tokens\n"
              "    - Save or write it down somewhere \n"
              "    - Type it in the prompt above\n\n"
              "'Repository not found' means you didn't give the token"
              " the correct permissions.\n"
              "'Authentication failed' means you mistyped your username or the"
              " token.\n")
        # We could prompt for username/password instead, but then the password
        # would be printed in plaintext if there is any problem during cloning.
        user = input('Gitlab username:')
        token = getpass.getpass('GitLab token: ')
    return user, token


def ensure_repo(repo_name, repo_path, user=None, token=None):
    """Clones private repository (prompting for credentials) if we do not have it"""
    if not os.path.exists(repo_path):
        print("Private data requested, we must clone repository folder.")
        user, token = ensure_token()
        temp_folder = ''.join(random.choices(string.ascii_lowercase, k=8))
        fd.run_command(f'git clone https://{user}:{token}'
                    f'@gitlab.com/{repo_name} {temp_folder}')
        fd.run_command(f'mv {temp_folder}/{repo_path} .')
        fd.run_command(f'rm -r -f {temp_folder}')


@export
def get_lz_file(data_file_name):
    """Return information from file in lz_private_data/...
    """
    ensure_repo('luxzeplin/stats/LZFlameFit.git', PATH)
    return fd.get_resource(f'{PATH}/{data_file_name}')
