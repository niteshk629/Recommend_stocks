"""Setup script for Stock Recommendation System"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-recommendation-system",
    version="1.0.0",
    author="Stock Analysis Team",
    author_email="",
    description="AI/ML-driven stock recommendation system for Indian markets (NSE/BSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niteshk629/Recommend_stocks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "stock-recommend=main:main",
        ],
    },
)
