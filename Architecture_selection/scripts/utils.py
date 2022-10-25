def compute_output_conv2d(input_shape, kernel_size, stride, padding, dilation):
    "Compute the output shape of a 2D convolution layer."
    if isinstance(input_shape, tuple) and len(input_shape)==2:
        return (int((input_shape[0] + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1),
            int((input_shape[1] + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1))
    else:
        return int((input_shape + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)