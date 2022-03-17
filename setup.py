import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lytomo_watershed",
    version="1.0.0",
    author="Mahdi Qezlou, Drew Newman, Gwen Rudie, Simeon Bird",
    author_email="mahdi.qezlou@email.ucr.edu",
    description="Detection of protoclusters in Ly-alpha tomography surveys",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahdiqezlou/LyTomo_Watershed",
    project_urls={
        "Bug Tracker": "https://github.com/mahdiqezlou/LyTomo_Watershed",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'mpi4py',
        ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
