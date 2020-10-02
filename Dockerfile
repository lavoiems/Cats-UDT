FROM python:3.7

ENV HOME="/home/workdir"

ADD requirement.txt $HOME/requirement.txt
RUN pip install --upgrade pip
RUN pip install -r $HOME/requirement.txt

WORKDIR $HOME
ADD src $HOME/src
