python train.py --dataset-root ../S-BSST265/ --training-patch-size 512 --lr 0.001 --epoch 10 --batchsize 5 --dataset-train-list train.txt --dataset-val-list val.txt --with-3-class-gt 1 --edge-width 5 --loss-type ce --enable-resnet-pretrain 1 --resume 0 --use-cuda 1