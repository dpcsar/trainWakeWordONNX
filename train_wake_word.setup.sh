#!/usr/bin/env bash
python3.10 -m venv venv
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
