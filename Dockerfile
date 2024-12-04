FROM ubuntu/python:3.12-24.04_stable
RUN pip install torch
RUN pip install tensortrade
RUN pip install yfinance

CMD ["/bin/bash"]
