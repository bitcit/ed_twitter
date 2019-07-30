# Memory optimized implementation of optim.adagrad

# [[ ADAGRAD implementation for SGD
# ARGS:
# - `opfunc` : a function that takes a single input (X), the point of
#                    evaluation, and returns f(X) and df/dX
# - `x` : the initial point
# - `state` : a table describing the state of the optimizer; after each
#                    call the state is modified
# - `state.learningRate` : learning rate
# - `state['paramVariance']` : vector of temporal variances of parameters
# - `state.weightDecay` : scalar that controls weight decay
# RETURN:
# - `x` : the new x vector
# - `f(x)` : the function, evaluated before the update
# ]]

import torch


def adagrad_mem(opfunc, x, config, state):
    # (0) get/update state
    if config is None and state is None:
        print('no state table, ADAGRAD initializing')

    config = config if config else {}
    state = state if state else config
    lr = config['learningRate'] if 'learningRate' in config else 1e-3
    lrd = config['learningRateDecay'] if 'learningRateDecay' in config else 0
    wd = config['weightDecay'] if 'weightDecay' in config else 0
    state['evalCounter'] = state['evalCounter'] if 'evalCounter' in state else 0
    nevals = state['evalCounter']

    # (1) evaluate f(x) and df/dx
    fx, dfdx = opfunc(x)

    # (2) weight decay with a single parameter
    if wd != 0:
        dfdx.add(wd, x)

    # (3) learning rate decay (annealing)
    clr = lr / (1 + nevals * lrd)

    # (4) parameter update with single or individual learning rates
    if not state['paramVariance']:
        state['paramVariance'] = torch.Tensor().typeAs(x).resizeAs(dfdx).zero()

    state['paramVariance'].addcmul(1, dfdx, dfdx)
    state['paramVariance'].add(1e-10)
    state['paramVariance'].sqrt()  # Keeps the std
    x.addcdiv(-clr, dfdx, state['paramVariance'])
    # Keeps the variance again
    state['paramVariance'].cmul(state['paramVariance'])
    state['paramVariance'].add(-1e-10)

    # (5) update evaluation counter
    state['evalCounter'] += 1

    # return x*, f(x) before optimization
    return x, {fx}
