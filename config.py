import numpy as np

NUM_WORKER = 32

IMAGE_HEIGHT=128
IMAGE_WIDTH =128

DEMO_HEIGHT = 128
DEMO_WIDTH  = 128

NUM_EPOCHS = 200

MAX_STEP = 1500000

LEARNING_RATE = 1e-4

MOMENTUM = 0.9

WEIGHT_DECAY = 1e-5

MILESTONES = [40, 60, 80]

DEVICE_IDS = [0]

BATCH_SIZE = 64

START_GRAD = 5

START_NORM = 7


###################################
MAE_GOAL = 0.4
USE_NETWORK_SLIMMING = True
L1_COEF = 1e-5
##################################


H, W = IMAGE_HEIGHT, IMAGE_WIDTH

H1, W1 = np.int(np.ceil(H/2)), np.int(np.ceil(W/2))

H2, W2 = np.int(np.ceil(H1/2)), np.int(np.ceil(W1/2))

H3, W3 = np.int(np.ceil(H2/2)), np.int(np.ceil(W2/2))

H4, W4 = np.int(np.ceil(H3/2)), np.int(np.ceil(W3/2))

H5, W5 = np.int(np.ceil(H4/2)), np.int(np.ceil(W4/2))
