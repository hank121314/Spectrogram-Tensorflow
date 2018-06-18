from glob import glob
import re

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown dog'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

def load_wav_file():
    possible = set(POSSIBLE_LABELS)
    fpaths = glob('./input/train/audio/*/*wav')
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    with open('./input/train/validation_list.txt', 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for fpath in validation_files:
        r = re.match(pattern, fpath)
        if r:
            valset.add(r.group(3))
    val,train=[],[]
    for fpath in fpaths:
        r = re.match(pattern, fpath)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, fpath)

            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val
