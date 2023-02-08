#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy
import petsc4py

def get_petsc_variables():
    """Get a list of PETSc environment variables from the file:
    $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables

    The result is memoized to avoid constantly reading the file.
    """
    import os
    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    path = [PETSC_DIR, PETSC_ARCH, "lib/petsc/conf/petscvariables"]
    variables_path = os.path.join(*path)
    with open(variables_path) as fh:
        # Split lines on first '=' (assignment)
        splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
    return {k.strip(): v.strip() for k, v in splitlines}

def configure():
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    PETSC_DIR  = os.environ['PETSC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),
                         join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib')]
    LIBRARIES += ['petsc']

    petscvariables = get_petsc_variables()
    os.environ['CC'] = petscvariables['CC']
#    os.environ['CFLAGS'] = petscvariables['CC_FLAGS']

    # PETSc for Python
    INCLUDE_DIRS += [petsc4py.get_include()]

    # NumPy
    INCLUDE_DIRS += [numpy.get_include()]

    return dict(
        include_dirs=INCLUDE_DIRS + [os.curdir],
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        runtime_library_dirs=LIBRARY_DIRS,
    )

extensions = [
Extension('petsclinearsystem', 
              sources = ['src/petsclinearsystem.pyx',  # key source c-python file, give numpy obj to c array
                         'src/petsclinearsystemimpl.c'], # key source c file, used to generate c file without impl
              depends = ['src/petsclinearsystem.h'],
              **configure()),
]

setup(name = "petsclinearsystem_XDiff", # package name
      version = '0.1',
       description = 'use petsc to solve HJB with cross-difference term',
       author = 'Bin Cheng',
       ext_modules = cythonize(
          extensions, include_path=[petsc4py.get_include()]),
       long_description = '''
Add also reflective boundary conditions.
''',
       author_email = 'bin.cheng.chicago@gmail.com',
)