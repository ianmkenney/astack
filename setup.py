from setuptools import setup

setup(name="astack",
      version="0.1",
      author="Ian Kenney",
      author_email="ianmichaelkenney@gmail.com",
      packages=["astack"],
      install_requires=["numpy",
                        "h5py",
                        "tqdm"]
      )
