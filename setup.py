from setuptools import setup, find_packages

setup(
    name='Kachas',
    version='0.1.0',
    description='A simple OCR library for Greek Polytonic texts',
    author='Konstantinos Perifanos',
    author_email='kostas.perifanos@gmail.com',
    url='https://www.github.com/kperi/kalchas',
    packages=find_packages(include=['Kalchas', 'Kalchas.*']),
    install_requires=[
        'numpy>=1.14.5',
        'torch>=2.2.0',
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['my-command=exampleproject.example:main']
    },
    package_data={'Kalchas': ['data/schema.json']}
)