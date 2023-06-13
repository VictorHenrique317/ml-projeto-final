FROM jupyter/scipy-notebook

COPY . /home/jovyan
USER jovyan
RUN jupyter nbconvert --to latex /home/jovyan/main.ipynb