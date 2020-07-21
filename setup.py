import setuptools

# Get requirements from requirements.txt
with open('requirements.txt') as f:
    requires = [x.strip() for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='flamedisx',
    version='1.2.0',
    description='Fast likelihood analysis in more dimensions for xenon TPCs',
    author='Jelle Aalbers, Bart Pelssers, Cristian Antochi',
    url='https://github.com/FlamTeam/flamedisx',
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=['pytest'],
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
