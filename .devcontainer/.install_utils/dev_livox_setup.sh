#! /bin/bash

apt-get update

pip install cython
# downloading a branch of a fork because they fixed python3 stuff in there
git clone --branch python3 https://github.com/pmusau17/range_libc.git
cd ./range_libc/pywrapper

python3 setup.py install

# install Livox-SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git /tmp/Livox-SDK2
cd /tmp/Livox-SDK2 && mkdir build && cd build
cmake .. && make -j$(nproc) && make install
rm -rf /tmp/Livox-SDK2
