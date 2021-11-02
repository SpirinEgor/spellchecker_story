#!/bin/bash

DATA_FOLDER="data"

# Create data folder
mkdir -p $DATA_FOLDER

# Download dictionaries for HunSpell from https://github.com/wooorm/dictionaries
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.dic -P $DATA_FOLDER
wget https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/en/index.aff -P $DATA_FOLDER

# Download test data
wget http://aspell.net/test/cur/batch0.tab -P $DATA_FOLDER

# Download train data
wget https://www.norvig.com/ngrams/spell-errors.txt -P $DATA_FOLDER
