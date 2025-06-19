import setuptools

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt", "r", encoding="utf-8") as req_file:
    # Temporarily remove requirements for faster development iteration on Kaggle.
    # Kaggle has torch, but for some reason if it's here, it spends a lot of time installing it.
    requirements = []  # [r.strip() for r in req_file.readlines()]

setuptools.setup(
    name="cayleypy",
    version="0.1.0",
    author="CayleyPy Foundation",
    description="Cayley graphs processing using GPU and pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/iKolt/cayleypy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "cayleypy": ["data/*.csv"],
    },
    python_requires=">=3.9",
    install_requires=requirements,
)
