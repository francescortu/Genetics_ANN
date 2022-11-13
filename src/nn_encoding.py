from src.mutations import *


class Net(nn.Module):
    def __init__(self, Net_encod):
        super().__init__()

        # verify the input shape and if net is not working update the encoding
        Net_encod.update_encoding()

        Net_encod.setting_channels()
        self.layer_list = []

        # initial input shape will be updated after each layer addition
        self.current_input_shape = Net_encod.get_input_shape()

        for i in range(Net_encod.len_features()):
            for j in range(Net_encod.GA_encoding(i).len()):
                layer = self.make_layer(Net_encod.GA_encoding(i).layers[j])
                self.layer_list.append(layer)

                # update current input shape
                self.current_input_shape = Net_encod.GA_encoding(i).layers[j].compute_shape(self.current_input_shape)

        self.layer_list.append(nn.Flatten())

        for i in range(Net_encod.len_classification()):
            for j in range(Net_encod.GA_encoding(Net_encod.len_features() + i).len()):
                self.layer_list.append(self.make_layer(Net_encod.GA_encoding(Net_encod.len_features() + i).layers[j]))

        self.layer_list.append(self.make_layer(Net_encod.last_layer[0].layers[0]) )
        self.layers = nn.Sequential(*self.layer_list)
        self.Net_encoding = Net_encod
        
        

    def make_layer(self, dsge_encod):
            if dsge_encod.type == layer_type.CONV:
                return nn.Conv2d(dsge_encod.channels['in'], dsge_encod.channels['out'], dsge_encod.param['kernel_size'], dsge_encod.param['stride'], dsge_encod.param['padding'], bias = dsge_encod.param['bias'])
            if dsge_encod.type == layer_type.LINEAR:
                    return nn.Linear(dsge_encod.channels['in'], dsge_encod.channels['out'])
            if dsge_encod.type == layer_type.ACTIVATION:
                if dsge_encod.param == activation.RELU:
                    return nn.ReLU()
                if dsge_encod.param == activation.SIGMOID:
                    return nn.Sigmoid()
                if dsge_encod.param == activation.TANH:
                    return nn.Tanh()
                if dsge_encod.param == activation.SOFTMAX:
                    return nn.Softmax()
            if dsge_encod.type == layer_type.POOLING:
                if dsge_encod.param["pool_type"] == pool.MAX:
                    return nn.MaxPool2d(dsge_encod.param['kernel_size'], dsge_encod.param['stride'], dsge_encod.param['padding'])
                elif dsge_encod.param["pool_type"] == pool.ADP_MAX:
                    return nn.AdaptiveMaxPool2d(self.current_input_shape)
                elif dsge_encod.param["pool_type"] == pool.AVG:
                    return nn.AvgPool2d(dsge_encod.param['kernel_size'], dsge_encod.param['stride'], dsge_encod.param['padding'])
                elif dsge_encod.param["pool_type"] == pool.ADP_AVG:
                    return nn.AdaptiveAvgPool2d(self.current_input_shape)
            if dsge_encod.type == layer_type.BATCH_NORM:
                    return nn.BatchNorm2d(dsge_encod.channels['in'])

    def forward(self, x):
        out = self.layers(x)
        return out
 
    def len(self):
        return len(self.layer_list)

    
    
