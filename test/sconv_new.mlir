module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
    %arg0: !transform.any_op) {
    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %res_list, %loops_list = transform.structured.sconv %convs
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) 
      -> (!transform.op<"linalg.generic">, !transform.any_op)
  
    // transform.foreach %res_list : !transform.op<"linalg.generic"> {
    //   // Insert additional transformations ...
    // }

    transform.yield
  }
}
