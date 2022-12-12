from setuptools import (
    setup,
    find_packages,
)


with open("README.md") as f:
    readme = f.read()

setup(
    name="Music Classification",
    version="0.1.0",
    description="Make use of ID3 algorithm to build a decision tree for categorising musics",
    long_description=readme,
    author="Nguyễn Khắc Trường",
    author_email="johnkt12qz@gmail.com",
    url="https://github.com/pseudoCode34/MusicClassification",
    packages=find_packages(exclude=("tests", "docs")),
)
