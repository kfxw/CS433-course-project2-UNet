train.py => Uses dice loss for half the training, then switches to BCE with HEM (weighted sum)
train2.py => Only uses a weighted sum of dice loss and HEM
train3.py => Only uses HEM