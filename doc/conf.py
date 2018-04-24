# -*- coding: utf-8 -*-

import datetime
import os
import re
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


def process_docstring(app, what, name, obj, options, lines):
    greek = dict(
        mu='μ',
        pi='π',
        sigma='σ',
        tau='τ',
    )

    for n, s in enumerate(lines):
        s = re.sub(r'\btau\b', greek['tau'], s)
        s = s.replace('f(0)(500)', 'f\ :sub:`0`\ (500)')

        for pattern in [
                r'(fm)(\^)([0-9-]+)',
                r'(pi|sigma)([_^])([a-z]+)',
                r'\b([pv])(\B)([xyz])\b'
        ]:
            s = re.sub(
                pattern,
                lambda m: '{}\ :{}:`{}`'.format(
                    greek.get(m[1], m[1]),
                    'sup' if m[2] == '^' else 'sub',
                    greek.get(m[3], m[3])
                ),
                s
            )

        if s != lines[n]:
            lines[n] = s


def setup(app):
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('autodoc-process-docstring', process_docstring)
