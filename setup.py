from setuptools import setup, find_packages

setup(
    name="elysia",
    version="0.1.0",
    packages=find_packages(include=['Project_Sophia', 'Project_Sophia.*', 'tests', 'tests.*']),
    install_requires=[
        # Dependencies from requirements.txt can be listed here
        # For now, this setup is primarily for making the project structure recognizable
    ],
    # This makes the project installable in editable mode,
    # which is key for resolving import issues during development and testing.
    entry_points={
        'console_scripts': [
            'elysia-bridge=applications.elysia_bridge:main',
        ],
    },
)
