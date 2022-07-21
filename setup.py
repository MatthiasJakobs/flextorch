import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name='flextorch',
      version='0.1',
      description='Extensible PyTorch wrapper for easy training',
      long_description=long_description,
      long_description_content_type='text/markdown',
      keywords=['machine learning', 'pytorch'],
      author='Matthias Jakobs',
      author_email='matthias.jakobs@tu-dortmund.de',
      url='https://github.com/MatthiasJakobs/flextorch',
      license='MIT',
      packages=setuptools.find_packages(),
      python_requires='>=3.8',
      install_requires=[])
