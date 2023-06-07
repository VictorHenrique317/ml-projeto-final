FROM jupyter/scipy-notebook

COPY . /home/jovyan
USER jovyan
RUN jupyter nbconvert --to script /home/jovyan/main.ipynb
RUN time python3 /home/jovyan/main.py