FROM python:latest
WORKDIR /usr/app/src
# RUN git clone https://github.com/telmo-correa/gym-snake.git && \
#     cd gym-snake && \
#     python -m pip install -e .
ADD requirements.txt /usr/app/src/
RUN python -m pip install -r requirements.txt