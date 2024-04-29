from distutils.core import setup
setup(
  name = 'alignment',
  packages = ['alignment'],
  version = '0.0.1',
  description = 'Aligns multiple cameras in 3d',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/SheffieldMLtracking/alignment.git',
  download_url = 'https://github.com/SheffieldMLtracking/alignment.git',
  keywords = ['registration','cameras','3d','pose estimation','position','orientation'],
  classifiers = [],
  install_requires=['numpy']
)
