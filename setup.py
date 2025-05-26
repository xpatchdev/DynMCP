from setuptools import setup, find_packages

setup(
    name="dynmcp",
    version="0.1.0",
    description="Dynamically expose class methods as MCP endpoints using FastMCP.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/dynmcp",
    packages=find_packages(include=["dynmcp", "dynmcp.*"]),
    install_requires=[
        "fastmcp",  # Make sure this is the correct package name for FastMCP
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
