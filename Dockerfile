FROM jupyter/scipy-notebook

COPY . /
RUN jupyter nbconvert --to script main.ipynb
RUN python3 main.py
