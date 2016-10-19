#! /bin/bash

#echo "test"




../cumf_sgd -g 1 -l 0.05 -a 0.08 -b 0.3 -u 1 -v 1 -x 1 -y 1 -s 768 -k 128 -t 20 /userdata/xiaolong/cmu_dataset/netflix_mm.bin

/home/xiaolong/programming/xx_libmf-2.01/mf-predict -e 0 /userdata/xiaolong/cmu_dataset/netflix_mme.bin netflix_mm.bin.model

rm -f netflix_mm.bin.model

