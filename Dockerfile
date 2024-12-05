FROM python:3.11-slim
RUN apt update
RUN apt install -y git
RUN apt install python3-tk tk-dev -y
WORKDIR /tensortrade
RUN pip install --no-cache-dir --upgrade pip
RUN pip install ray[rllib]
RUN pip install torch
RUN pip install git+https://github.com/tensortrade-org/tensortrade.git
RUN pip install yfinance

CMD ["/bin/bash"]
