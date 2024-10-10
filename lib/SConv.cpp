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
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "sconv-transform"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

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

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  if (isa<IntegerType>(x.getType()))
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (isa<IntegerType>(accType))
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

// Implementation of SConv transform dialect operation.
DiagnosedSilenceableFailure
transform::SConvOp::applyToOne(transform::TransformRewriter &rewriter,
                               LinalgOp namedOp,
                               transform::ApplyToEachResultList &results,
                               transform::TransformState &state) {

  // 0. Startup
  Location loc = namedOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  rewriter.setInsertionPoint(namedOp); /* Is it necessary? */

  // 1. Rewrite the named operation as a generic.
  auto genericOp = dyn_cast<GenericOp>(namedOp.getOperation());
  if (!genericOp) {
    FailureOr<GenericOp> generalizeResult =
        generalizeNamedOp(rewriter, namedOp);
    assert(succeeded(generalizeResult) && "unexpected failure generalizing op");
    genericOp = *generalizeResult;
  }

  // 2. Replace the affine maps, iterator types and output tensor shape
  SmallVector<Value> inputs = genericOp.getDpsInputs();
  auto inputType = cast<ShapedType>(genericOp.getInputs()[0].getType());
  auto filterType = cast<ShapedType>(genericOp.getInputs()[1].getType());
  auto outputType = cast<ShapedType>(genericOp.getOutputs()[0].getType());

  // if (!filterType.hasStaticShape())
  //   return rewriter.notifyMatchFailure(genericOp,
  //                                      "expected a static shape for the filter");

  // if (!inputType.hasStaticShape())
  //   return rewriter.notifyMatchFailure(genericOp,
  //                                      "expected a static shape for the input");

  Value input = genericOp.getInputs()[0];
  Value output = genericOp.getOutputs()[0];

  auto outputShape = outputType.getShape();
  int64_t n = outputShape[0];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];

  // 3. Insert the Collapse shape before the newOp
  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());

  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

  AffineExpr d0, d1, d3, d4, d5, d6;
  bindDims(context, d0, d1, d3, d4, d5, d6);
  auto d7 = d3.floorDiv(oh) + d5;
  auto d8 = d3 % oh + d6;
  auto lhsMap = AffineMap::get(4, 0, {d0, d4, d7, d8}, context);
  auto rhsMap = AffineMap::get(4, 0, {d1, d4, d5, d6}, context);
  auto resultMap = AffineMap::get(3, 0, {d0, d1, d3}, context);

  // 4. Create the new generic Op
  auto newOp = rewriter.create<linalg::GenericOp>(
      loc,
      reshapedOutputType,
      inputs,
      ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
      newOpIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });

  Value result = newOp.getResults().front();

  // 5. Insert the Expanded Shape after the newOp
  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(newOp, ArrayRef<Value>{reshapedResult});

  // 6. Call the CSA Analysis
  
  // 7. Call the mlir::transform::TileUsingForOp (...) twice

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
