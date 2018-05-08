STATE_WINDOW = 1
CONVS = []
FCS = [64, 64]
PADDING = 'VALID'
LSTM_UNIT = 64
FINAL_STEP = 10 ** 6
ACTORS = 8

LSTM = True
POLICY_FACTOR = 1.0
VALUE_FACTOR = 0.5
ENTROPY_FACTOR = 0.01
LR = 1e-4
LR_DECAY = 'constant'
GRAD_CLIP = 40.0
TIME_HORIZON = 5
GAMMA = 0.99
