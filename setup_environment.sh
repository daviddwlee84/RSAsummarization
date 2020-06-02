#!/bin/bash

# https://pip.pypa.io/en/stable/installing/
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python2 get-pip.py
rm get-pip.py

sudo apt-get install python2.7-dev

pip install -r requirements.txt --user
