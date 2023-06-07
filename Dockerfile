FROM jupyter/scipy-notebook

COPY . /home/jovyan
RUN /home/jovyan/run_all.sh
RUN time python3 /home/jovyan/main.py
