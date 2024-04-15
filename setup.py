from setuptools import find_packages, setup

setup(
    name = "MedAR",
    version = "0.1.0",
    author = "Mou YongLi, Ailin Liu, Moritz Busch, Diego Collarana, Sulayman Sowe, Stefan Decker",
    author_email = "mou@dbis.rwth-aachen.de",
    description = ("Medical Abbreviation Resolution via Knowledge Enhanced Tranformer"),
    license = "MIT",
    url = "https://github.com/MouYongli/MedAR",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Medical Image Segmentation",
        "License :: OSI Approved :: MIT License",
    ],
)