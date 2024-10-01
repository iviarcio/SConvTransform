//===-- SConv.cpp - Transform dialect Extension SConv ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the SConv.
//
//===----------------------------------------------------------------------===//

#include "SConv.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

// Define a new transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class SConv
    : public ::mlir::transform::TransformDialectExtension<SConv> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SConv)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void SConv::init() {
  // Similarly to dialects, an extension can declare a dependent dialect. This
  // dialect will be loaded along with the extension and, therefore, along with
  // the Transform dialect. Only declare as dependent the dialects that contain
  // the attributes or types used by transform operations. Do NOT declare as
  // dependent the dialects produced during the transformation.
  // declareDependentDialect<MyDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  // In the following chapter, we will be add operations that generate function
  // calls and structured control flow operations, so let's declare the
  // corresponding dialects as generated.
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SConv.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "SConv.cpp.inc"

// Implementation of SConv transform dialect operation.
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          TransformResults &transformResults,
                          TransformState &state) {

  // Call CSA
  ArrayRef<int64_t> tileSizes_1 = [1, 64, 1, 32, 16, 0, 0];   // 16 canais , 32 colunas, 64 filtros, 2o, 1 tile de uma linha
  ArrayRef<int64_t> tileSizes_2 = [0, 8, 0, 16, 0, 0, 0];   // 16 canais , 32 colunas, 64 filtros, 2o, 1 tile de uma linha

  ArrayRef<int64_t> interchange_1 = [0, 4, 3, 2, 1]; // 4 = F, 3: OH, 2:OW, 1:C
  ArrayRef<int64_t> interchange_2 = [1, 0];

  // mlir::transform::GeneralizeOp (...)
  // affine_map<(n, oc, oh, ow) -> (n, oc, oh*ow)>
  // mlir::transform::TileUsingForOp (...)
  // mlir::transform::TileUsingForOp (...)
  // mlir::transform::VectorizeOp (...)
  // mlir::transform::OneShotBufferizeOp (...)

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::SConvOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {

  // Indicate that the payload is modified by this operation.
  modifiesPayload(effects);
}

LogicalResult mlir::transform::SConvOp::verify() {
  return success();
}
void registerSConv(::mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
