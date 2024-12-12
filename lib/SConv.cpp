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

#include "CSA.h"

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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
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
#include <cstdint>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

#define DEBUG_TYPE "sconv-transform"
#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "]: ")

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

// Delinearizes the given composite `index` by the basis specified in `factors`.
static SmallVector<Value> unrollIndex(OpBuilder &b, Location loc, Value index,
                                      ArrayRef<int64_t> factors) {
  assert(!factors.empty() && "empty factor list");
  SmallVector<Value> basis;
  for (int64_t f : factors)
    basis.push_back(b.create<arith::ConstantOp>(loc, b.getIndexAttr(f)));
  FailureOr<SmallVector<Value>> multiIndex =
      affine::delinearizeIndex(b, loc, index, basis);
  assert(!failed(multiIndex) && "Failed to linearize sconv index");
  return *multiIndex;
}

// Given indices corresponding to iterators in the output (oIndex) and filter
// (fIndex) for a convolution, compute the convolved index for the input as
// `oIndex * stride + fIndex`.
static Value getConvolvedIndex(OpBuilder &b, Location loc, Value oIndex,
                               Value fIndex, int64_t stride) {
  AffineExpr oExpr, fExpr, strideExpr;
  bindDims(b.getContext(), oExpr, fExpr);
  strideExpr = b.getAffineConstantExpr(stride);
  
  // Create the affine map with both indices as inputs.
  AffineMap convMap = AffineMap::get(2, 0, oExpr * strideExpr + fExpr, b.getContext());
  
  // Apply the affine map to the provided indices.
  return affine::makeComposedAffineApply(b, loc, convMap, {oIndex, fIndex});
}

// After the inner tile operation, promote the two affine.apply and the first extracted
// slice if chd = IS or the second extracted slice if schd = WS, to the outer loop
static LogicalResult
promoteOpsOfTile(RewriterBase &rewriter, Operation *transformOp,
                 CSAStrategy res, SmallVector<Operation *> loopOps) {

  // Cast to scf::ForOp the  outermost loop of the second tile
  int i = 0;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[i]);
  if (!outerLoop) return transformOp->emitError("failed to get the outermost scf::for op");

  // Get the body of the outer loop
  Block *outerBody = outerLoop.getBody();

  // Locate the inner loop within the outer loop's body
  scf::ForOp innerLoop;
  for (Operation &op : outerBody->getOperations()) {
    if (auto loop = dyn_cast<scf::ForOp>(&op)) {
      innerLoop = loop;
      break;
    }
  }
  if (!innerLoop) return transformOp->emitError("failed to get the inner forOp");

  // Get the body of the inner loop
  Block *innerBody = innerLoop.getBody();

  Operation *affineApply1 = nullptr;
  Operation *affineApply2 = nullptr;
  Operation *tensorExtractSlice1 = nullptr;
  Operation *tensorExtractSlice2 = nullptr;

  // Locate the target operations within the inner loop body
  for (Operation &op : innerBody->getOperations()) {
    if (!affineApply1 && isa<AffineApplyOp>(&op))
      affineApply1 = &op;
    else if (!affineApply2 && isa<AffineApplyOp>(&op))
      affineApply2 = &op;
    else if (!tensorExtractSlice1 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice1 = &op;
    else if (!tensorExtractSlice2 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice2 = &op;

    // Stop once all target operations are found
    if (affineApply1 && affineApply2 && tensorExtractSlice1 && tensorExtractSlice2)
      break;
  }

  if (!affineApply1 || !affineApply2 || !tensorExtractSlice1 || !tensorExtractSlice2)
    return transformOp->emitError("Failed to hosting Ops");

  // Set insertion point at the start of outer loop
  rewriter.setInsertionPointToStart(outerBody);

  Operation *promotedAffineApply1 = nullptr;
  Operation *promotedAffineApply2 = nullptr;
  Operation *promotedTensorExtractSlice1 = nullptr;
  Operation *promotedTensorExtractSlice2 = nullptr;

  // Clone operations to the outer loop
  if (res.schd == IS) {
    promotedAffineApply1 = rewriter.clone(*affineApply1);
    promotedAffineApply2 = rewriter.clone(*affineApply2);
    promotedTensorExtractSlice1 = rewriter.clone(*tensorExtractSlice1);
  }
  else {
    promotedTensorExtractSlice2 = rewriter.clone(*tensorExtractSlice2);
  }

  // Replace uses of the old operations in the inner loop body
  innerBody->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      if (res.schd == IS && operand.get() == affineApply1->getResult(0))
        operand.set(promotedAffineApply1->getResult(0));
      else if (res.schd == IS && operand.get() == affineApply2->getResult(0))
        operand.set(promotedAffineApply2->getResult(0));
      else if (res.schd == IS && operand.get() == tensorExtractSlice1->getResult(0))
        operand.set(promotedTensorExtractSlice1->getResult(0));
      else if (res.schd == WS && operand.get() == tensorExtractSlice2->getResult(0))
        operand.set(promotedTensorExtractSlice2->getResult(0));
    }
  });

  // Erase the old operations
  if (affineApply1->use_empty()) rewriter.eraseOp(affineApply1);
  if (affineApply2->use_empty()) rewriter.eraseOp(affineApply2);
  if (tensorExtractSlice1->use_empty()) rewriter.eraseOp(tensorExtractSlice1);
  if (tensorExtractSlice2->use_empty()) rewriter.eraseOp(tensorExtractSlice2);

  return success();
}

// Apply the filter packing. This packing will be inserted at the begining of first or
// second loop level of the internal convolution depends of Input or Wheight Stationary
static LogicalResult
applyFilterPacking(RewriterBase &rewriter, Operation *transformOp, CSA csa, CSAStrategy res,
                  SmallVector<Operation *> tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  // select the loop based on IS or WS
  int loopIndex = res.schd == IS ? 1 : 0;

  // Cast to scf::ForOp the  selected loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the inner (generic) convOp
  auto convOp = tiledOps.front();

  // Cast to linalg::GenericOp to get the input & filter, types & shapes
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the inner convOp");

  // Locate the insertion point (scf::for Op if WS or linalg::generic Op if IS)
  Location loc = linalgOp.getLoc(); // location should never be null.
  if (res.schd == WS) {
    scf::ForOp innerLoop;
    for (Operation &op : loopBody->getOperations()) {
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        innerLoop = loop;
        break;
      }
    }
    if (!innerLoop) return transformOp->emitError("failed to get the inner forOp");
    rewriter.setInsertionPoint(innerLoop);
    loc = innerLoop->getLoc();
  }
  else {
    rewriter.setInsertionPoint(linalgOp);
  }

  SmallVector<Value> inputs = linalgOp.getInputs();
  Value input = inputs[0];
  Value filter = inputs[1];

  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());
  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();

  // Compute the Packed Filter Shape: {ic × fh × fw, nf}
  int64_t ic = inputShape[1];
  int64_t nf = filterShape[0];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
  SmallVector<int64_t, 2> filterPackingShape = {ic * fh * fw, nf};
  Value filterPacking = rewriter.create<tensor::EmptyOp>(loc, filterPackingShape, filterType.getElementType());

  auto nloops = filterPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 2> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 3> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto packingTensor = rewriter.create<linalg::GenericOp>(
    loc, filterPacking.getType(),
    /*inputs=*/ValueRange{},
    /*outputs=*/filterPacking,
    /*indeexingMaps=*/ packingIndexingMaps,
    /*iteratorTypes=*/ packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      // Get the iterators
      Value index0 = nestedBuilder.create<linalg::IndexOp>(loc, 0); // ic × fh × fw
      Value index1 = nestedBuilder.create<linalg::IndexOp>(loc, 1); // nf

      // Unroll index0 into {ic, fh, fw}
      SmallVector<Value> unpackedIndices = unrollIndex(
          nestedBuilder, nestedLoc, index0, ArrayRef<int64_t>{ic, fh, fw});
      Value icIndex = unpackedIndices[0];
      Value fhIndex = unpackedIndices[1];
      Value fwIndex = unpackedIndices[2];

      // Create the extraction indices for the original filter tensor
      SmallVector<Value> extractionIndices{index1, icIndex, fhIndex, fwIndex};

      // Extract the value from the original filter tensor
      Value filterVal = nestedBuilder.create<tensor::ExtractOp>(loc, filter, extractionIndices);

      // Yield the extracted value into the packed tensor
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, filterVal);
    });

  return success();
}

// Apply the input packing. This packing will be inserted at the begining of first or
// second loop level of the internal convolution depends of Input or Wheight Stationary
static LogicalResult
applyInputPacking(RewriterBase &rewriter, Operation *transformOp, CSA csa,
                  CSAStrategy res, SmallVector<int64_t, 2> strides,
                  SmallVector<Operation *> tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  // select the loop based on IS or WS
  int loopIndex = res.schd == IS ? 0 : 1;

  // Cast to scf::ForOp the  selected loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the inner (generic) convOp
  auto convOp = tiledOps.front();

  // Cast to linalg::GenericOp to get the input & filter, types & shapes
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the inner convOp");

  // Locate the insertion point (scf::for Op if IS or linalg::generic Op if WS)
  Location loc = linalgOp.getLoc(); // location should never be null.
  if (res.schd == IS) {
    scf::ForOp innerLoop;
    for (Operation &op : loopBody->getOperations()) {
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        innerLoop = loop;
        break;
      }
    }
    if (!innerLoop) return transformOp->emitError("failed to get the inner forOp");
    rewriter.setInsertionPoint(innerLoop);
    loc = innerLoop->getLoc();
  }
  else {
    rewriter.setInsertionPoint(linalgOp);
  }

  SmallVector<Value> inputs = linalgOp.getInputs();
  SmallVector<Value> outputs = linalgOp.getOutputs();
  Value input = inputs[0];
  Value filter = inputs[1];
  Value output = outputs[0];

  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();

  auto outputType = cast<ShapedType>(output.getType());
  auto outputShape = outputType.getShape();

  int64_t n = inputShape[0];
  int64_t ic = inputShape[1];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
  int64_t oh = outputShape[1];
  int64_t ow = outputShape[2];
  int64_t nw = csa.mK_.nwindows;

  // Compute the Packed Input Shape: {n, ic × fh × fw, nw}
  SmallVector<int64_t, 3> inputPackingShape = {n, ic * fh * fw, nw};
  Value inputPacking = rewriter.create<tensor::EmptyOp>(loc, inputPackingShape, inputType.getElementType());

  auto nloops = inputPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 3> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 3> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto packingTensor = rewriter.create<linalg::GenericOp>(
    loc, inputPacking.getType(),
    /*inputs=*/ValueRange{}, /*outputs=*/inputPacking, packingIndexingMaps,
    packingIterators,

    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      // Get the iterators
      Value index0 = nestedBuilder.create<linalg::IndexOp>(loc, 0); // n
      Value index1 = nestedBuilder.create<linalg::IndexOp>(loc, 1); // ic × fh × fw
      Value index2 = nestedBuilder.create<linalg::IndexOp>(loc, 2); // nw 

      // Unroll index1 into {ic, fh, fw}
      SmallVector<Value> unpackedSpatialIndices = unrollIndex(
          nestedBuilder, nestedLoc, index1, ArrayRef<int64_t>{ic, fh, fw});
      Value icIndex = unpackedSpatialIndices[0];
      Value fhIndex = unpackedSpatialIndices[1];
      Value fwIndex = unpackedSpatialIndices[2];

      // Compute the actual spatial indices using strides
      Value wIndex = getConvolvedIndex(nestedBuilder, nestedLoc, index2, fwIndex, strides[1]);

      // Create the extraction indices for the original input tensor
      SmallVector<Value> extractionIndices{index0, icIndex, fhIndex, wIndex};

      // Extract the value from the original input tensor
      Value inputVal = nestedBuilder.create<tensor::ExtractOp>(loc, input, extractionIndices);

      // Yield the extracted value into the packed tensor
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
    });

  return success();
}

// Apply a tiling transformation to a modified payload ops and store both the
// tiled operation as well as the created tile loops.
static LogicalResult
applyTileTo(RewriterBase &rewriter, Operation *transformOp, Operation *target,
            CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, 
            transform::TransformResults &transformResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Assign tile sizes
  // Input Stationary: N, NF * K2, NWIN * K3, NC, FH, FW  
  // Weight Stationary: N, NF * K3, NWIN * K2, NC, FH, FW  
  int64_t nFTiles = csa.mK_.num_filters * (res.schd == IS ? res.k2 : res.k3);
  int64_t nWinTiles = csa.mK_.nwindows * (res.schd == IS ? res.k3 : res.k2);
  SmallVector<int64_t, 6> tileSize = {1, nFTiles, nWinTiles, res.tile_c, 0, 0};
  SmallVector<OpFoldResult> tileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), tileSize);

  // Order:
  // Input Stationary: N, NC, NWIN, NF
  // Weight Stationary: N, NC, NF, NWIN
  int64_t nFOrder = res.schd == IS ? 3 : 2;
  int64_t nWinOrder = res.schd == IS ? 2 : 3;
  SmallVector<int64_t, 4> tileInterchange = {0, nFOrder, nWinOrder, 1};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOfr).setInterchange(tileInterchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);
  if (failed(tiledResults))
    return transformOp->emitError("failed the outermost tile operation");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(tilingInterfaceOp, tiledResults->replacements);

  // Perform the tiling in the inner convolution
  auto innerOp = tiledResults->tiledOps.front();

  SmallVector<int64_t, 6> innerTileSize = {0, csa.mK_.num_filters, csa.mK_.nwindows, 0, 0, 0};
  SmallVector<OpFoldResult> innerTileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), innerTileSize);

  int64_t innerFOrder = res.schd == IS ? 1 : 0;
  int64_t innerWinOrder = res.schd == IS ? 0 : 1;
  SmallVector<int64_t, 2> innerInterchange = {innerFOrder, innerWinOrder};

  auto innerTilingInterfaceOp = dyn_cast<TilingInterface>(innerOp);
  if (!innerTilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");
  scf::SCFTilingOptions innerTilingOptions;
  innerTilingOptions.setTileSizes(innerTileSizesOfr).setInterchange(innerInterchange);
  innerTilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(innerOp);
  FailureOr<scf::SCFTilingResult> innerTiledResults =
      scf::tileUsingSCF(rewriter, innerTilingInterfaceOp, innerTilingOptions);
  if (failed(innerTiledResults))
    return transformOp->emitError("failed the innermost tile operation");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(innerTilingInterfaceOp, innerTiledResults->replacements);

  // Report back the relevant handles to the transform op.
  tiledOps.push_back(innerTiledResults->tiledOps.front());
  for (Operation *loop : innerTiledResults->loops)
    loopOps.push_back(loop);
  for (Operation *loop : tiledResults->loops)
    loopOps.push_back(loop);

  // Promote some inner loop Ops depends of schedule (WS or IS)
  LogicalResult result0 = promoteOpsOfTile(rewriter, transformOp, res, loopOps);
  if (failed(result0)) return transformOp->emitError("failed to hosting Ops");

  // Generate the filter packing
  LogicalResult result1 = applyFilterPacking(rewriter, transformOp, csa, res, tiledOps, loopOps);
  if (failed(result1)) return transformOp->emitError("failed to apply the filter packing");

  // Generate the input packing
  LogicalResult result2 = applyInputPacking(rewriter, transformOp, csa, res, strides, tiledOps, loopOps);
  if (failed(result2)) return transformOp->emitError("failed to apply the input packing");

  transformResults.set(transformOp->getOpResult(0), tiledOps);
  for (auto [index, loop] : llvm::enumerate(loopOps))
    transformResults.set(transformOp->getOpResult(index + 1), {loop});

  return success();
}

///
/// Implementation of SConv::apply transform dialect operation.
///
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Get context and convOp params
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  assert(llvm::hasSingleElement(targetOps) && "expected a single target op");

  auto convOp = dyn_cast_or_null<linalg::Conv2DNchwFchwOp>(*targetOps.begin());
  if (!convOp)  
    return emitSilenceableError() << "expected a Conv2DNchwFchwOp for transformation";

  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  SmallVector<Value> inputs = convOp.getDpsInputs();
  ValueRange outputs = convOp.getDpsInits();
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
  if (!hasAllOneValues(convOp.getDilations()))
    return emitSilenceableError() << "expected all ones for dilations";

  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();
  auto outputShape = outputType.getShape();
  int64_t n = outputShape[0];
  int64_t ic = inputShape[1];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
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
  auto hstride = convOp.getStrides().getValues<int64_t>()[0];
  auto wstride = convOp.getStrides().getValues<int64_t>()[1];
  SmallVector<int64_t, 2> strides = {hstride, wstride};

  AffineExpr d0, d1, d2, d3, d4, d5;
  bindDims(context, d0, d1, d2, d3, d4, d5);
  auto lhsMap = AffineMap::get(6, 0, {d0, d3, d2.floorDiv(oh) * hstride + d4, d2 % oh * wstride + d5}, context);
  auto rhsMap = AffineMap::get(6, 0, {d1, d3, d4, d5}, context);
  auto resultMap = AffineMap::get(6, 0, {d0, d1, d2}, context);

  // Create the new genericOp that replaces the named convolution
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

  // Create the Expanded Shape
  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(loc, outputType, genericOp.getResults().front(), outputReassocIndices);

  // replace convOp with (reshapedOutput + genericOp + reshapedResult)
  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  // Call the CSA Analysis
  ConvInfo csaConv = {ic, oh, ow, fh, fw, oc, 4};
  CSA csa = createCSAPass(csaConv);
  CSAStrategy res = csa();
  
  /* Just for test */
  res.schd = WS; res.k2 = 2; res.k3 = 8; res.tile_c = 16;
  /* Comment the code above to use the CSA Analysis */

  // Apply the tile in the genericOp based on the CSA Analysis
  LogicalResult result = applyTileTo(rewriter, getOperation(), genericOp, csa, res, strides, results);

  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

void transform::SConvOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
