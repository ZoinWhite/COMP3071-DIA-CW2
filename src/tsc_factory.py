from src.trafficsignalcontrollers.uniformcycletsc import UniformCycleTSC
from src.trafficsignalcontrollers.nextdurationrltsc import NextDurationRLTSC
from src.rl_factory import rl_factory

def tsc_factory(tsc_type, tl, args, netdata, rl_stats, exp_replay, neural_network, eps, conn):
    if tsc_type == 'uniform':
        return UniformCycleTSC(conn, tl, args.mode, netdata, args.r, args.y, args.g_min)
    elif tsc_type == 'ddpg':
        ddpgagent = rl_factory(tsc_type, args,
                                neural_network, exp_replay, rl_stats, 1, eps)
        return NextDurationRLTSC(conn, tl, args.mode, netdata, args.r, args.y,
                                 args.g_min, args.g_max, ddpgagent)
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc)+' does not exist.'
