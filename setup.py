from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="simple-collection",
    version="0.0.1",
    author="AJ Stangl",
    author_email="ajstangl@gmail.com",
    description="Code for my damn self",
    license="MIT",
    keywords="GPT2",
    include_package_data=True,
    url="https://example.com",
    packages=[
        'common',
        'common/captioning',
        'common/progress',
        'common/schemas',
        'common/storage',
        'common/functions'
    ],

    long_description=read('README.md'),
    classifiers=[
        "Topic :: Utilities",
        "License :: MIT License",
    ],
    entry_points={
        'console_scripts': [
            'simple-collection = main:main.py',
        ],
    },
    requires=['adlfs', 'praw']
)
