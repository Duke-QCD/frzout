# -*- coding: utf-8 -*-

import os
import sys

import frzout

sys.path.insert(0, os.path.abspath(os.curdir))

project = 'frzout'
version = release = frzout.__version__
author = 'Jonah Bernhard'
copyright = '2016 ' + author

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

default_role = 'math'
pygments_style = 'sphinx'

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_context = dict(
    display_github=True,
    github_user='jbernhard',
    github_repo='frzout',
    github_version='master',
    conf_py_path='/doc/',
    source_suffix = source_suffix,
)

html_domain_indices = False
html_use_index = False
