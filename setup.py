from setuptools import setup, find_packages

setup(
    name="jigsaw_toxic_comment_challenge",
    version="0.1.0",
    description="Jigsaw Toxic Comment Classification Challenge: A solution to the Kaggle competition",
    author="marsolmos",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
