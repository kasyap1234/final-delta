#!/usr/bin/env python3
"""
Delta Exchange India Trading Bot - Setup Script

This script sets up the trading bot package for installation.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="delta-trading-bot",
    version="1.0.0",
    author="Trading Bot Developer",
    author_email="developer@example.com",
    description="A sophisticated cryptocurrency trading bot for Delta Exchange India",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/delta-trading-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ccxt>=4.0.0",
        "websockets>=11.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "colorama>=0.4.6",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "aiosqlite>=0.19.0",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "logging": [
            "structlog>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "delta-bot=bot.trading_bot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.sql", "*.json"],
    },
    zip_safe=False,
)