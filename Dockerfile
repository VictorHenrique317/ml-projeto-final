FROM jupyter/scipy-notebook

COPY . /home/jovyan
RUN chmod a+x /home/jovyan/run_all.sh
RUN /home/jovyan/run_all.sh
RUN time python3 /home/jovyan/main.py
