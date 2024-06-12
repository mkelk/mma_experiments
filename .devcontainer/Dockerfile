# [Choice] Python version (use -bookworm or -bullseye variants on local arm64/Apple Silicon): 3, 3.12, 3.11, 3.10, 3.9, 3.8, 3-bookworm, 3.12-bookworm, 3.11-bookworm, 3.10-bookworm, 3.9-bookworm, 3.8-bookworm, 3-bullseye, 3.12-bullseye, 3.11-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3-buster, 3.12-buster, 3.11-buster, 3.10-buster, 3.9-buster, 3.8-buster
# ARG VARIANT=3.12-bookworm
FROM python:3.12

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common 

# Set the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/workspaces/BookCode_Edition1_me/"

# Temporary: Upgrade python packages due to https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-40897
# They are installed by the base image (python) which does not have the patch.
# RUN python3 -m pip install --upgrade setuptools

# use poetry
RUN pip install poetry==1.5.1

# install graphviz
RUN apt-get install -y graphviz

