module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    //===------------------------------------------------------------------===//
    // APPLY SCONV
    //===------------------------------------------------------------------===//

    %convs = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %ukernels_0, %loops = transform.structured.sconv %convs
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) 
      -> (!transform.op<"linalg.generic">, !transform.any_op)

    //===------------------------------------------------------------------===//
    // BUFFERIZE
    //===------------------------------------------------------------------===//

    %bufferized = transform.bufferization.one_shot_bufferize %arg0
      { bufferize_function_boundaries = true }
      : (!transform.any_op) -> !transform.any_op

    %funcs = transform.structured.match ops{["func.func"]} in %bufferized 
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %funcs {
      transform.apply_patterns.memref.extract_address_computations 
      transform.apply_patterns.memref.expand_strided_metadata 
    } : !transform.any_op

    //===------------------------------------------------------------------===//
    // SIMPLIFY
    //===------------------------------------------------------------------===//

    // Apply common sub-expression elimination
    transform.apply_cse to %bufferized : !transform.any_op
    // Apply dead-code elimination
    transform.apply_dce to %bufferized : !transform.any_op

    //===------------------------------------------------------------------===//
    // LOWER
    //===------------------------------------------------------------------===//

    // Get another handle to the kernels because the last one was invalidated
    // by one-shot bufferize.
    %ukernels_1 = transform.structured.match
      ops{["linalg.generic"]}
      attributes {microkernel} in %bufferized
      : (!transform.any_op) -> !transform.op<"linalg.generic">
 
    // Lower each microkernel individually
    transform.foreach %ukernels_1 : !transform.op<"linalg.generic"> -> !transform.any_op {
    ^bb1(%op: !transform.op<"linalg.generic">):
      %lowered = transform.lower.to_blas "sgemm_blas_kernel", %op
        : (!transform.op<"linalg.generic">) -> (!transform.any_op)
      transform.yield %lowered : !transform.any_op
    }

    transform.yield
  }
}

// vim: sts=2 sw=2 et
