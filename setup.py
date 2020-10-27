#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='visual_gtsam',
      version='0.1',
      description='Installation of GTSAM and basic usage',
      author='WareVision LLC Team',
      author_email='',
      package_dir={},
      packages=["visual_gtsam", "visual_gtsam.dataset", "visual_gtsam.dataset.structures",
                "visual_gtsam.barcode_detector",
                "visual_gtsam.barcode_detector.utils"],
      install_requires=install_requires
      )
