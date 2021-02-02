from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# General
__C.MODEL_NAME = 'tmp'
__C.WEIGHTS_PATH = ''

# Logging
__C.LOG = CN()
__C.LOG.SAVE_ITER = 2000
__C.LOG.SAMPLED_FIXED_NOISE = False
__C.LOG.SAVE_SINGLES = False
__C.LOG.LOSSES = False

# Model
__C.MODEL = CN()
__C.MODEL.RES = 256
__C.MODEL.TRUNCATION = 1.0

# Training
__C.TRAIN = CN()
__C.TRAIN.EPISODES = 50
__C.TRAIN.BATCH_SZ = 256
__C.TRAIN.BATCH_ACC = 8

# Loss Weigths
__C.LAMBDA = CN()
__C.LAMBDA.L1 = 100
__C.LAMBDA.PERC = [4, 4, 4, 4]
__C.LAMBDA.BINARY = 300
__C.LAMBDA.MASK = 500
__C.LAMBDA.TEXT = [0, 5, 5, 0]
__C.LAMBDA.BG = 2000

# Learning Rates
__C.LR = CN()
__C.LR.SHAPE = 8e-6
__C.LR.TEXTURE = 3e-5
__C.LR.BG = 1e-5

def get_cfg_defaults():
    return __C.clone()
