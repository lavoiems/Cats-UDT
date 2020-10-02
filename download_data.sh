#!/bin/bash

FOLDER=data

if [[ $1  =~ ^(sketch|real)$ ]];
then
    echo "Downloading data"
    wget http://csr.bu.edu/ftp/visda/2019/multi-source/$1.zip
    mkdir $FOLDER
    mv $1.zip $FOLDER
    cd $FOLDER
    unzip $1.zip "$1/bird/*"
    unzip $1.zip "$1/dog/*"
    unzip $1.zip "$1/flower/*"
    unzip $1.zip "$1/speedboat/*"
    unzip $1.zip "$1/tiger/*"

    #rm -rf $1.zip
    #cd $1

    mkdir train
    mkdir train/$1
    mkdir train/$1/bird
    mkdir train/$1/dog
    mkdir train/$1/flower
    mkdir train/$1/speedboat
    mkdir train/$1/tiger

    mkdir test
    mkdir test/$1
    mkdir test_all
    mkdir test_all/$1
    mkdir test/$1/bird
    mkdir test/$1/dog
    mkdir test/$1/flower
    mkdir test/$1/speedboat
    mkdir test/$1/tiger

    mkdir fid
    mkdir fid/$1
    mkdir fid/$1/all
    mkdir fid/$1/bird
    mkdir fid/$1/dog
    mkdir fid/$1/flower
    mkdir fid/$1/speedboat
    mkdir fid/$1/tiger

    cp $1/bird/* fid/$1/all
    cp $1/bird/* fid/$1/bird
    cp $1/dog/* fid/$1/all
    cp $1/dog/* fid/$1/dog
    cp $1/flower/* fid/$1/all
    cp $1/flower/* fid/$1/flower
    cp $1/speedboat/* fid/$1/all
    cp $1/speedboat/* fid/$1/speedboat
    cp $1/tiger/* fid/$1/all
    cp $1/tiger/* fid/$1/tiger

    ls $1/bird/      | head -15 | xargs -i cp $1/bird/{}      test_all/$1
    ls $1/bird/      | head -15 | xargs -i mv $1/bird/{}      test/$1/bird
    ls $1/dog/       | head -15 | xargs -i cp $1/dog/{}       test_all/$1
    ls $1/dog/       | head -15 | xargs -i mv $1/dog/{}       test/$1/dog
    ls $1/flower/    | head -15 | xargs -i cp $1/flower/{}    test_all/$1
    ls $1/flower/    | head -15 | xargs -i mv $1/flower/{}    test/$1/flower
    ls $1/speedboat/ | head -15 | xargs -i cp $1/speedboat/{} test_all/$1
    ls $1/speedboat/ | head -15 | xargs -i mv $1/speedboat/{} test/$1/speedboat
    ls $1/tiger/     | head -15 | xargs -i cp $1/tiger/{}     test_all/$1
    ls $1/tiger/     | head -15 | xargs -i mv $1/tiger/{}     test/$1/tiger

    ls $1/bird/      | xargs -i mv $1/bird/{}      train/$1/bird
    ls $1/dog/       | xargs -i mv $1/dog/{}       train/$1/dog
    ls $1/flower/    | xargs -i mv $1/flower/{}    train/$1/flower
    ls $1/speedboat/ | xargs -i mv $1/speedboat/{} train/$1/speedboat
    ls $1/tiger/     | xargs -i mv $1/tiger/{}     train/$1/tiger

    rm -rf $1

else
    echo "$1 is not a valid choice."
fi

