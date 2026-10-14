"""
SignLingo — Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="signlingo",
    version="1.0.0",
    description="Multi-language sign language recognition and sentence formation",
    author="Ishika",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.12.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "signlingo-train=scripts.train:main",
            "signlingo-eval=scripts.evaluate:main",
            "signlingo-bench=scripts.benchmark:main",
            "signlingo-demo=scripts.demo_webcam:main",
        ],
    },
)
