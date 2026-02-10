"""QuantForge-Toolkit: Professional Quantitative Finance Framework."""

from setuptools import setup, find_packages

setup(
    name="quantforge-toolkit",
    version="1.0.0",
    description="Quantitative Finance Toolkit: Trading, Optimization & Sentiment",
    author="QuantForge Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "vectorbt>=0.26.0",
        "riskfolio-lib>=4.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
    ],
)
