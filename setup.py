import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="verse_monster",
    version='0.0.1',
    url="git@github.com:kevinbache/verse_monster",
    license='MIT',

    author="Kevin Bache",
    author_email="kevin.bache@gmail.com",

    description="He eats rhythm and spits rhyme",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    # When you install this package, this console_scripts option ensures that another executable python script named
    # `verse_monster_docker_entrypoint` is added to your path which will call the function called
    # `main_funk` in the file imported in
    # `verse_monster.verse_monster_docker_entrypoint`
    # entry_points={
    #     'console_scripts': [
    #         'verse_monster_docker_entrypoint=verse_monster.verse_monster_docker_entrypoint:main_funk',
    #     ],
    # },

    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
        'transformers',
        'datasets',
        'tokenizers',
        'sacrebleu',
        'scikit-learn',
        'cloudpickle',
        'torchtext',
        'torch',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

