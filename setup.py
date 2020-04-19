import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="owopclient",
    version="0.0.1",
    author="Will Vesey",
    author_email="will@vesey.tech",
    description="A custom client to connect to Our World Of Pixels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1ttric/owopclient",
    install_requires=["asyncio", "aiohttp", "aiohttp-socks", "Pillow", "numpy"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
