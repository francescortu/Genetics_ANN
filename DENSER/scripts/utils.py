def compute_output_conv2d(input_shape, kernel_size, stride, padding, dilation=1):
    "Compute the output shape of a 2D convolution layer."

    if padding == "same":
        return input_shape

    elif padding == "valid":   #se il padding è valid allora devo calcolare la dimensione dell'output
        padding = 0
    return int((input_shape + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

def compute_output_avgpool2d(input_shape, kernel_size, stride, padding, dilation=1):
    "Compute the output shape of a 2D average pooling layer."
    return int((input_shape + 2*padding - (kernel_size) )/stride + 1)



def compute_input_conv2d(output_shape, kernel_size, stride, padding, dilation=1):
    "Compute the input shape of a 2D convolution layer."
    if isinstance(output_shape, tuple) and len(output_shape)==2:
        return (int((output_shape[0] - 1)*stride[0] + dilation[0]*(kernel_size[0]-1) - 2*padding[0] + 1),
            int((output_shape[1] - 1)*stride[1] + dilation[1]*(kernel_size[1]-1) - 2*padding[1] + 1))
    else:
        return int((output_shape - 1)*stride + dilation*(kernel_size-1) - 2*padding + 1)

def compute_padding_same_max_pool2d(input_shape, out_shape, kernel_size, stride, dilatation=1):
    padding = (out_shape -1) * stride - input_shape + dilatation * (kernel_size - 1) +1 
    return int(padding/2)

def compute_padding_same_avg_pool2d(input_shape, out_shape, kernel_size, stride, dilatation=1):
    padding = (out_shape -1) * stride - input_shape + kernel_size 
    return int(padding/2)
