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

  // 1. Get context and namedOp params
  MLIRContext *context = rewriter.getContext();
  Location loc = namedOp.getLoc();

  SmallVector<Value> inputs = namedOp.getDpsInputs();
  ValueRange outputs = namedOp.getDpsInits();
  Value input = inputs[0];
  Value filter = inputs[1];
  Value output = outputs[0];

  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());
  auto outputType = cast<ShapedType>(output.getType());

  auto outputShape = outputType.getShape();
  int64_t n = outputShape[0];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];

  // 2. Create the Collapse shape to be inserted at begining
  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  // 3. Create the affine maps, iterator types and output tensor shape
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

  AffineExpr d0, d1, d3, d4, d5, d6;
  bindDims(context, d0, d1, d3, d4, d5, d6);
  auto lhsMap = AffineMap::get(6, 0, {d0, d4, d3.floorDiv(oh) + d5, d3 % oh + d6}, context);
  auto rhsMap = AffineMap::get(6, 0, {d1, d4, d5, d6}, context);
  auto resultMap = AffineMap::get(6, 0, {d0, d1, d3}, context);

  // 4. Create the new generic Op
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc,
      reshapedOutputType,
      inputs,
      ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
      newOpIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul = createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });

  Value result = genericOp.getResults().front();

  // 5. Create the Expanded Shape to be inserted after the genericOp
  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  // 6. Replace the namedOp to genericOp
  rewriter.replaceOp(namedOp, ArrayRef<Value>{reshapedResult});

  // 6. Call the CSA Analysis
  
  // 7. Call the mlir::transform::TileUsingForOp (...) twice

  // If everything went well, return success.

  results.push_back(genericOp);

  return DiagnosedSilenceableFailure::success();
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
