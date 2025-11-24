"""
SAM 3D Body - Pip-installable Package
Setup configuration for building and distributing the package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="sam-3d-body",
    version="1.0.0",
    author="Meta Superintelligence Labs",
    author_email="",
    description="SAM 3D Body: Robust Full-Body Human Mesh Recovery from single images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/sam-3d-body",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "pyrender>=0.1.45",
        "opencv-python>=4.8.0",
        "yacs>=0.1.8",
        "scikit-image>=0.21.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "dill>=0.3.7",
        "pandas>=2.0.0",
        "rich>=13.0.0",
        "hydra-core>=1.3.0",
        "hydra-submitit-launcher>=1.2.0",
        "hydra-colorlog>=1.2.0",
        "pyrootutils>=1.0.4",
        "webdataset>=0.2.0",
        "chump>=1.6.0",
        "networkx==3.2.1",
        "roma>=1.3.0",
        "joblib>=1.3.0",
        "seaborn>=0.12.0",
        "wandb>=0.15.0",
        "appdirs>=1.4.4",
        "ffmpeg-python>=0.2.0",
        "cython>=3.0.0",
        "jsonlines>=4.0.0",
        "pytest>=7.4.0",
        "xtcocotools>=1.14",
        "loguru>=0.7.0",
        "optree>=0.9.0",
        "fvcore>=0.1.5",
        "black>=23.0.0",
        "pycocotools>=2.0.7",
        "tensorboard>=2.14.0",
        "huggingface-hub>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "detectron2": [
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
        ],
        "moge": [
            "moge @ git+https://github.com/microsoft/MoGe.git",
        ],
    },
    entry_points={
        "console_scripts": [
            "sam3d-demo=sam_3d_body.cli:demo_cli",
            "sam3d-inference=sam_3d_body.cli:inference_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "sam_3d_body": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    zip_safe=False,
)
