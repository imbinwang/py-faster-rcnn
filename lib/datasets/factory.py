# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.linemod_ape
import datasets.linemod_sub
import datasets.linemod
import datasets.pascal_voc
import numpy as np

### my own dataset ###

#------ real images for training and testing ------
linemod_devkit_path = '/mnt/wb/dataset/LINEMOD4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod', split)
    __sets[name] = (lambda split=split: datasets.linemod(split, linemod_devkit_path))
#------ real images ------

#------ synthesized images for training ------
linemod_largesub_devkit_path = '/mnt/wb/dataset/LINEMOD_LARGE4FRCNN'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_largesub', split)
    __sets[name] = (lambda split=split: datasets.linemod_sub(split, linemod_largesub_devkit_path))
#------ synthesized images ------

#------ synthesized images for training ------
linemod_largeape_devkit_path = '/mnt/wb/dataset/LINEMOD_APE_PARAM/BG_PARAM'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_largeape', split)
    __sets[name] = (lambda split=split: datasets.linemod_ape(split, linemod_largeape_devkit_path))
#------ synthesized images ------

#------ synthesized images for training ------
linemod_largeapeb_devkit_path = '/mnt/wb/dataset/LINEMOD_APE_PARAM/BG_PARAM_B'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_largeapeb', split)
    __sets[name] = (lambda split=split: datasets.linemod_ape(split, linemod_largeapeb_devkit_path))
#------ synthesized image --------

#------ synthesized images for training ------
linemod_largeapec_devkit_path = '/mnt/wb/dataset/LINEMOD_APE_PARAM/BG_PARAM_C'
for split in ['train', 'test']:
    name = '{}_{}'.format('linemod_largeapec', split)
    __sets[name] = (lambda split=split: datasets.linemod_ape(split, linemod_largeapec_devkit_path))
#------ synthesized image --------

### my own dataset ###

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
