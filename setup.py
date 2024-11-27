from setuptools import setup, find_packages

setup(
    name="transformers_gad",
    version="0.1",
    packages=[
        "transformers_gad",
        "transformers_gad.generation",
        "transformers_gad.build_oracle"
    ],
    install_requires=[
        # Add any dependencies here
        "torch",
        "transformers",
        "tqdm",
        # Add other dependencies from requirements.txt if needed
    ],
)