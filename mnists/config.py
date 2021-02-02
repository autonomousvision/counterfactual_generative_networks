from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# General
__C.MODEL_NAME = 'tmp'

# Logging
__C.LOG = CN()
__C.LOG.SAVE_ITER = 1000
__C.LOG.LOSSES = True

# Model
__C.MODEL = CN()
__C.MODEL.N_CLASSES = 10
__C.MODEL.LATENT_SZ = 12
__C.MODEL.NDF = 12
__C.MODEL.NGF = 12
__C.MODEL.DISC = 'linear'
__C.MODEL.INIT_TYPE = 'orthogonal'
__C.MODEL.INIT_GAIN = 0.5

# Training
__C.TRAIN = CN()
__C.TRAIN.DATASET = 'colored_MNIST'
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.WORKERS = 12
__C.TRAIN.COLOR_VAR = 0.02
__C.TRAIN.EPOCHS = 5

# Loss Weigths
__C.LAMBDAS = CN()
__C.LAMBDAS.MASK = 0.5
__C.LAMBDAS.PERC = [0, 0.05, 0.05, 0]

# Learnign Rates
__C.LR = CN()
__C.LR.LR = 2e-4
__C.LR.BETAS = [0.5, 0.999]

def get_cfg_defaults():
    return __C.clone()
