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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "sconv-transform"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

#define GET_OP_CLASSES
#include "SConv.cpp.inc"

// Define the SConv transform dialect. This uses the CRTP idiom to identify extensions.
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

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

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

// Implementation of SConv::apply transform dialect operation.
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          linalg::Conv2DNchwFchwOp namedOp,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Get context and namedOp params
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

  if (!filterType.hasStaticShape())
    return emitSilenceableError() << "expected a static shape for the filter";

  if (!inputType.hasStaticShape())
    return emitSilenceableError() << "expected a static shape for the input";

  // Does not support dilation.
  if (!hasAllOneValues(namedOp.getDilations()))
    return emitSilenceableError() << "expected all ones for dilations";

  auto outputShape = outputType.getShape();
  int64_t n = outputShape[0];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];

  // Create the Collapse shape to be inserted at begining
  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  // Create the affine maps, iterator types and output tensor shape
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

  // Get strides
  auto hstride = namedOp.getStrides().getValues<int64_t>()[0];
  auto wstride = namedOp.getStrides().getValues<int64_t>()[1];

  AffineExpr d0, d1, d2, d3, d4, d5;
  bindDims(context, d0, d1, d2, d3, d4, d5);
  auto lhsMap = AffineMap::get(6, 0, {d0, d3, d2.floorDiv(oh) * hstride + d4, d2 % oh * wstride + d5}, context);
  auto rhsMap = AffineMap::get(6, 0, {d1, d3, d4, d5}, context);
  auto resultMap = AffineMap::get(6, 0, {d0, d1, d2}, context);

  // Create the new genericOp that replaces the namedOp
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

  Value res = genericOp.getResults().front();

  // Create the Expanded Shape to be inserted after the genericOp
  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(loc, outputType, res, outputReassocIndices);

  // Replace the namedOp to genericOp
  rewriter.replaceOp(namedOp, ArrayRef<Value>{reshapedResult});

  // TODO: Call the CSA Analysis
  
  // For now, define the tile sizes and interchange as constants
  SmallVector<int64_t, 6> tileSize = {1, 64, 32, 16, 0, 0};
  SmallVector<int64_t, 4> tileInterchange = {0, 3, 2, 1};

  // Create attributes for static tile sizes and interchange
  auto tileSizeAttr = rewriter.getDenseI64ArrayAttr(tileSize);
  auto interchangeAttr = rewriter.getDenseI64ArrayAttr(tileInterchange);

  // Set the insertion point in genericOp
  rewriter.setInsertionPointAfter(genericOp);

  // Create TileUsingForOp using the builder
  auto tiledOp = rewriter.create<transform::TileUsingForOp>(
    rewriter.getUnknownLoc(),          // Location  
    genericOp.getResult(0),            // The operation to tile ?
    tileSizeAttr,                      // Static tile sizes
    interchangeAttr);                  // Interchange attribute

  // Apply the tiling transformation using apply method
  DiagnosedSilenceableFailure status =  tiledOp.apply(rewriter, results, state); 
  if (!status.succeeded())
    return status;

  // Substitui a operação original
  rewriter.replaceOp(genericOp, tiledOp.getResults());

  // Remove a operação original ?
  rewriter.eraseOp(genericOp);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {
    return emitSilenceableError() << "expected a conv2d nchw convolution";
}

void transform::SConvOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
