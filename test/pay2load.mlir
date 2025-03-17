!input_tensor_t = tensor<1x96x56x56xf32>
!weight_tensor_t = tensor<24x96x1x1xf32>
!output_tensor_t = tensor<1x24x56x56xf32>
!weight2_tensor_t = tensor<144x24x1x1xf32>
!out2put_tensor_t = tensor<1x144x56x56xf32>

func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t, %wei2: !weight2_tensor_t,
                             %out1: !output_tensor_t, %out2: !out2put_tensor_t) -> !out2put_tensor_t {
  // First Convolution, strides = 1
  %res1 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out1: !output_tensor_t) -> !output_tensor_t

  // Second Convolution, strides = 1
  %res2 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%res1, %wei2: !output_tensor_t, !weight2_tensor_t)
    outs(%out2: !out2put_tensor_t) -> !out2put_tensor_t

  return %res2 : !out2put_tensor_t
}
