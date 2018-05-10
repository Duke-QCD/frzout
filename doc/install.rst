Installation
============

Requirements:

- Python 3.4+ (2.7 might work but is untested; Python 3 strongly recommended)
- numpy and scipy (0.18.0+)
- frzout has a compiled C extension, so a C compiler is required.
  On some systems, the Python development package may need to be explicitly installed, e.g. python3-dev on Ubuntu.

Install using pip::

   pip install frzout

Development version
-------------------
Building a development version (e.g. when modifying the code) additionally requires Cython.
After cloning `the git repository <https://github.com/Duke-QCD/frzout>`_, run ::

   pip install -e .

which installs in "`development mode <https://packaging.python.org/tutorials/distributing-packages/#working-in-development-mode>`_".
After modifying the Cython source code, re-run the command to recompile.
