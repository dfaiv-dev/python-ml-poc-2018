FROM pcsml/base

WORKDIR /opt/pcsml/py-scikit-spike
COPY src .

VOLUME /var/opt/pcsml/training-data
VOLUME /var/opt/pcsml/remote-exec-out

# by default, run without $DISPLAY
ENV MPLBACKEND=agg
ENV PYTHONPATH=/opt/pcsml/py-scikit-spike:$PYTHONPATH