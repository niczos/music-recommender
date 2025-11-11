from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]



# Read requirements from requirements.txt
requirements = parse_requirements('requirements.txt')

setup(
    name='XYZ',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your project',
    url='https://github.com/niczos/music-recommender-system',  # if you have a GitHub repo
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
