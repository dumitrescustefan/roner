import setuptools
def parse_requirements(filename, session=None):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roner",
    version="1.0.3",
    author="Stefan Daniel Dumitrescu",
    author_email="dumitrescu.stefan@gmail.com",
    description="Named Entity Recognition for Romanian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dumitrescustefan/roner",
    packages=setuptools.find_packages(),
    install_requires = parse_requirements('requirements.txt', session=False),
    include_package_data=True,
    data_files=[],
    entry_points={},
    classifiers=(
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
    ),
)
