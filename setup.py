from setuptools import setup, find_packages

VERSION = '1.0'
DESCRIPTION = 'Software infrastructure to for machine learning'
LONG_DESCRIPTION = 'Software infrastructure for machine learning projects that makes it easier to manage experiments and log progress'

setup(
    name="machine_learning_infrastructure",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Ada L",
    author_email="the.nostra.tymus@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='machine learning',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)