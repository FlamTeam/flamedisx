import setuptools

with open('requirements_minimal.txt') as f:
    requires = [x.strip() for x in f.readlines()]

with open('requirements.txt') as f:
    requires_strict = [x.strip() for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='flamedisx',
    version='2.0.0',
    description='Fast likelihood analysis in more dimensions for xenon TPCs',
    author='Flamedisx developers',
    url='https://github.com/FlamTeam/flamedisx',
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        'flamedisx': ['nest/config/*.ini'],
    },
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=['pytest'],
    extras_require={
        'docs': ['sphinx',
                 'sphinx_rtd_theme',
                 'nbsphinx',
                 'recommonmark'],
        'strict-deps': requires_strict,
    },
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'],
    zip_safe=False)
