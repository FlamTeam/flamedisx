"""Tools to get private LZ data
"""
import getpass
import os
import os.path
import random
import string
import sys

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


def clone_repo(repo_name, repo_path, user=None, token=None):
    """Clones private repository (prompting for credentials)"""
    fd.run_command(f'rm -r -f {repo_path}')
    user, token = ensure_token()
    temp_folder = ''.join(random.choices(string.ascii_lowercase, k=8))
    fd.run_command(f'git clone https://{user}:{token}'
                    f'@gitlab.com/{repo_name} {temp_folder}')
    fd.run_command(f'mv {temp_folder}/{repo_path} .')
    fd.run_command(f'rm -r -f {temp_folder}')


@export
def lz_setup(mode='tensors', run='sr1', lz_private_data_dir=None):
    """Update the system path to include necessary LZ private data files
    """
    if lz_private_data_dir is not None:
        if os.path.isdir(f'{lz_private_data_dir}/{mode}/{run}'):
            sys.path.append(f'{lz_private_data_dir}/{mode}/{run}')
            print(f'Successfully found mode {mode} and run {run} in ' + \
                  f'directory {lz_private_data_dir}')
            return
        else:
            print(f'Error finding mode {mode} and run {run} in directory {lz_private_data_dir}')

    print('Seeing if LZ private data directory already exists')

    if os.path.isdir(f'lz_private_data/{mode}/{run}'):
        sys.path.append(f'lz_private_data/{mode}/{run}')
        print(f'Successfully found mode {mode} and run {run} in directory lz_private_data')
        return
    else:
        print(f'Mode {mode} and run {run} not found in any existing directory')

    print('Cloning repo, requires LZ GitLab access token')

    clone_repo('luxzeplin/stats/LZFlameFit.git', PATH)

    if os.path.isdir(f'lz_private_data/{mode}/{run}'):
        sys.path.append(f'lz_private_data/{mode}/{run}')
        print(f'Successfully found mode {mode} and run {run} in cloned directory')
    else:
        raise RuntimeError(f'Either the cloning failed, or could not find ' + \
                           f'mode {mode} and run {run} in cloned directory')
