FROM jupyter/base-notebook
USER root
RUN apt-get update && apt-get -y install libgeos-dev libglib2.0-0
USER jovyan
RUN pip install shapely===1.6.4
RUN pip install tqdm===4.39.0
RUN pip install numpy===1.17.4
RUN pip install matplotlib===3.1.1
RUN pip install scikit-image===0.15.0
RUN pip install scikit-learn===0.21.3
RUN pip install ipywidgets===7.5.1
RUN pip install opencv-python===4.1.2.30
USER root
RUN apt-get install -y libsm6 libxext6
RUN apt-get -y install libxrender1
USER jovyan
RUN pip install pandas===0.25.3
RUN pip install svgpath2mpl
