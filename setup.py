from setuptools import setup, find_packages
import subprocess
import os

version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

if "-" in version:
    # when not on tag, git describe outputs: "1.3.3-22-gdf81228"
    # pip has gotten strict with version numbers
    # so change it to: "1.3.3+22.git.gdf81228"
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v, i, s = version.split("-")
    version = v + "+" + i + ".git." + s

assert "-" not in version
assert "." in version

assert os.path.isfile("version.py")
with open("VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % version)

# reading long description from file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# some more details
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
]

# calling the setup function
setup(
    name='stable-diffusion-pytorch',
    version=version,
    description='A Python library for stable diffusion model using PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/deepak7376/stable_diffusion_pytorch',
    author='Deepak Yadav',
    author_email='dky.united@gmail.com',
    license='MIT',
    py_modules=["stable-diffusion-pytorch"],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='stable diffusion pytorch generative modeling',
    python_requires='>=3',
    install_requires=[
        'numpy==1.26.2',
        'Pillow==10.1.0',
        'pytorch-lightning==2.1.3',
        'torch==2.1.2',
        'tqdm==4.66.1',
        'transformers==4.36.2'
        # Add other dependencies from requirements.txt
    ],
)
