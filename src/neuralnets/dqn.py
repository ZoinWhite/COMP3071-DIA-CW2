import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

from src.neuralnet import NeuralNet
from src.helper_funcs import check_and_make_dir

class DQN(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, learner=False):
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner)
        for model in self.models:
            self.models[model].compile(Adam(learning_rate=lr, epsilon=lre), loss='mse')

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        model_in = Input((input_d,))
        for i in range(len(hidden_d)):
            if i == 0:
                model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model_in)
            else:
                model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model)

        model_out = Dense(output_d, activation=output_act, kernel_initializer='he_uniform')(model)
        return Model(model_in, model_out)

    def forward(self, _input, nettype):
        return self.models[nettype].predict(_input)
  
    def backward(self, _input, _target):
        self.models['online'].fit(_input, _target, batch_size = 1, epochs = 1,  verbose=0 )
def load_weights(self, path):
    path += '.h5'
    if os.path.exists(path):
        self.models['online'].load_weights(path)
        # 只有当target存在时才更新它
        if 'target' in self.models:
            self.set_weights(self.get_weights('online'), 'target')
    else:
        assert 0, 'Failed to load weights, supplied weight file path '+str(path)+' does not exist.'

    def get_weights(self, nettype):
        return self.models[nettype].get_weights()
                                                          
    def set_weights(self, weights, nettype):
        return self.models[nettype].set_weights(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        self.models[nettype].save_weights(path+fname+'.h5', save_format='h5', overwrite='True')
       
    def load_weights(self, path):
        path += '.h5'
        print(f"正在尝试加载模型: {path}")
        if os.path.exists(path):
            try:
                print(f"模型文件存在，开始加载...")
                self.models['online'].load_weights(path)
                print(f"模型加载成功！")
                # 将权重也加载到target模型
                self.set_weights(self.get_weights('online'), 'target')
            except Exception as e:
                print(f"加载模型时出错: {str(e)}")
                print("尝试使用备选方法加载...")
                try:
                    # 尝试使用低级API加载
                    import h5py
                    with h5py.File(path, 'r') as f:
                        weight_names = [n.decode('utf8') if isinstance(n, bytes) else n 
                                       for n in f.attrs['weight_names']]
                        weights = [f[weight_name][()] for weight_name in weight_names]
                    
                    # 手动设置权重
                    self.models['online'].set_weights(weights)
                    print(f"使用备选方法加载成功！")
                    # 同步到target模型
                    self.set_weights(self.get_weights('online'), 'target')
                except Exception as e2:
                    print(f"备选方法也失败: {str(e2)}")
                    raise e  # 重新抛出原始错误
        else:
            # 提供详细的错误信息
            print(f"错误: 找不到模型文件 '{path}'")
            assert 0, f'Failed to load weights, supplied weight file path {path} does not exist.'

if __name__ == '__main__':
    input_d = 10
    dqn = DQN( input_d, [20, 20], 'relu', 4, 'linear', 0.0001, 0.0000001)

    x = np.random.uniform(0.0, 1.0, size=(1,input_d))

    ouptut = dqn.forward(x, 'online')
    print(output)
