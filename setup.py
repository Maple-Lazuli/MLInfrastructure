from setuptools import setup

VERSION = '1.8.1'
DESCRIPTION = 'Software infrastructure to for machine learning'
LONG_DESCRIPTION = 'Software infrastructure for machine learning projects that makes it easier to manage experiments and log progress'

setup(
    name="ml_infrastructure",
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    author="Ada L",
    author_email="the.nostra.tymus@gmail.com",
    license='MIT',
    packages=['ml_infrastructure'],
    install_requires=[
        'flask',
        'turbo_flask',
        'plotly',
        'requests',
        'torch',
        'numpy'
    ],
    package_data={'ml_infrastructure': ['templates/*', 'static/styles/*', 'static/js/*']},
    keywords='machine learning',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
)
