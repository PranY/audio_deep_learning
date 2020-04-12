from setuptools import setup, find_packages

setup(
    name="audio_deep_learning",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.wav', '*.pth']
    },
    entry_points={
        "console_scripts": ['inference=assignment.inference:main']
    }
)