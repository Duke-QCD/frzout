# -*- coding: utf-8 -*-

import datetime
import os
import sys

import frzout

sys.path.insert(0, os.path.abspath(os.curdir))

project = 'frzout'
version = release = frzout.__version__
author = 'Jonah Bernhard'
copyright = '{} {}'.format(datetime.date.today().year, author)

source_suffix = '.rst'
master_doc = 'index'

templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = ['_build']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'plot_tests',
]

plot_formats = [('png', 200), 'pdf']

default_role = 'any'

html_theme = 'sphinx_rtd_theme'
html_context = dict(show_source=False)
