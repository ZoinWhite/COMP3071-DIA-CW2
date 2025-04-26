import os, sys, time
import warnings

# 抑制警告消息
warnings.filterwarnings('ignore')
# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 抑制OpenMP信息输出
os.environ['KMP_WARNINGS'] = 'off'

from src.argparse import parse_cl_args
from src.distprocs import DistProcs

def main():
    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()
    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)
    print('...finish running main')
    print('run time '+str((time.time()-start_t)/60))

if __name__ == '__main__':
    main()
