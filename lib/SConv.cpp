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

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
// Define a new transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class SConv
    : public transform::TransformDialectExtension<SConv> {
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
  // As an transform extension dialect, we must declare all dependent dialects.
  // These dialects will be loaded along with the extension and, therefore,
  // along with the Transform dialect. The dependent dialects contain the
  // attributes or types used by transform operations.
  declareDependentDialect<linalg::LinalgDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SConv.cpp.inc"
      >();
}
} // namespace

#define GET_OP_CLASSES
#include "SConv.cpp.inc"

// Implementation of SConv transform dialect operation.
DiagnosedSilenceableFailure
transform::SConvOp::applyToOne(transform::TransformRewriter &rewriter,
                               LinalgOp target,
                               transform::ApplyToEachResultList &results,
                               transform::TransformState &state) {

  // 0. Startup
  rewriter.setInsertionPoint(target);

  // 1. Rewrite the named operation as a generic.
  auto genericOp = dyn_cast<GenericOp>(target.getOperation());
  if (!genericOp) {
    FailureOr<GenericOp> generalizeResult =
        generalizeNamedOp(rewriter, target);
    assert(succeeded(generalizeResult) && "unexpected failure generalizing op");
    genericOp = *generalizeResult;
  }

  // 2. Get the current affine maps, iterator types and output tensor shape
  SmallVector<Value> inputs = genericOp.getDpsInputs();
  ValueRange outputs = genericOp.getDpsInits();
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators = genericOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = genericOp.hasPureTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};

  // 3. Replace the affine maps, iterator types and output tensor shape
  // TOD

  GenericOp newOp = rewriter.create<GenericOp>(
      genericOp.getLoc(), resultTypes, inputs, outputs, indexingMaps, iterators);
  rewriter.inlineRegionBefore(genericOp->getRegion(0), newOp.getRegion(),
                              newOp.getRegion().begin());
  rewriter.replaceOp(genericOp, newOp->getResults());

  // 4. Insert the Collapse shape and Expanded Shape before and after the newOp
  // TOD

  // 5. Call the CSA Analysis
  
  // 6. Call the mlir::transform::TileUsingForOp (...)

  // 7. Call the mlir::transform::TileUsingForOp (...)

  // If everything went well, return success.
  return DiagnosedSilenceableFailure::success();
}

// void transform::SConvOp::getEffects(
//     ::llvm::SmallVectorImpl<::MemoryEffects::EffectInstance> &effects) {
//
//   // Indicate that the payload is modified by this operation.
//   modifiesPayload(effects);
// }

// LogicalResult transform::SConvOp::verify() {
//   return success();
// }

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
