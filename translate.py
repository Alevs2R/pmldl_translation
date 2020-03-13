# download and unpack model from https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz

import sys
import numpy as np
import string
import random
import os
import time

def get_lines(file):
    reader = open(file)
    train_lines = reader.readlines()
    reader.close()

    return train_lines

lines = get_lines('test.ru.txt') 

from fairseq.models.transformer import TransformerModel
ru2en = TransformerModel.from_pretrained(
  '/home/aleksei/Documents/wmt19.ru-en.ensemble',
  checkpoint_file='model1.pt'
)

result_lines = []
for line in lines:
    translated = ru2en.translate(line)
    print(translated)
    result_lines.append(translated)

print('translated, saving to file...')

with open('output.txt', 'a') as the_file:
    for line in result_lines:
        the_file.write('%s\n' % line)

print('saved')
