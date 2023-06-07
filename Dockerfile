FROM jupyter/scipy-notebook

COPY . /home/jovyan
RUN jupyter nbconvert --to script /home/jovyan/main.ipynb
RUN python3 /home/jovyan/main.py
