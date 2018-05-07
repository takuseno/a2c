STATE_SHAPE = [84, 84]
STATE_WINDOW = 1
CONVS = [[32, 3, 2], [32, 3, 2], [32, 3, 2], [32, 3, 2]]
FCS = []
PADDING = 'SAME'
LSTM_UNIT = 256
FINAL_STEP = 10 ** 8
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
