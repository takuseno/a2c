# A2C
A2C imeplementation with TensorFlow.

## requirements
- Python3

## dependencies
- tensorflow
- opencv-python
- numpy
- git+https://github.com/imai-laboratory/rlsaber
- git+https://github.com/openai/baselines

## train
```
$ python train.py [--render] [--env environment id]
```

## play
```
$ python train.py --demo [--render] [--load {path to model}] [--env environment id]
```

## implementations
This repostory is inspired by following projects.

- [OpenAI Baselines](https://github.com/openai/baselines)
- [A3C](https://github.com/takuseno/a3c)
