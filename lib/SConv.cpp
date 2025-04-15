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
#include "mlir/Transforms/DialectConversion.h"
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
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

// Compute the multi-packing filter indices considering strides and tensor structure.
static Value computeFilterIndices(OpBuilder &b, Location loc, Value oIndex,
                               Value fIndex, int64_t stride) {
  AffineExpr oExpr, fExpr, strideExpr;
  bindDims(b.getContext(), oExpr, fExpr);
  strideExpr = b.getAffineConstantExpr(stride);
  
  // Create the affine map with both indices as inputs.
  AffineMap convMap = AffineMap::get(2, 0, {oExpr * strideExpr + fExpr}, b.getContext());
  
  // Apply the affine map to the provided indices.
  return affine::makeComposedAffineApply(b, loc, convMap, {oIndex, fIndex});
}

// Compute the linearized input indices considering strides and tensor structure.
static SmallVector<Value, 2> computeLinearInputIndices(
    OpBuilder &b, Location loc, Value kIndex, Value nwinIndex, Value IOin,
    Value IOout, Value ILss, int64_t Fh, int64_t Fw, int64_t Ow, int64_t Iw, 
    SmallVector<int64_t, 2> strides) {

  MLIRContext *context = b.getContext();

  int64_t filterHW = Fh * Fw;
  int64_t strideH = strides[0];
  int64_t strideW = strides[1];

  // Compute NcIndex (iNc) = iK floorDiv (Fh * Fw)
  AffineExpr d0;
  bindDims(context, d0);
  AffineMap NcMap = AffineMap::get(1, 0, {d0.floorDiv(filterHW)}, context);
  Value NcIndex = affine::makeComposedAffineApply(b, loc, NcMap, {kIndex});

  // Compute ILstart = IOin + IOout + ILss
  AffineExpr d1, d2, s0;
  bindDims(context, d1, d2);
  bindSymbols(context, s0);
  AffineMap ILstartMap = AffineMap::get(2, 1, {d1 + d2 + s0}, context);
  Value ILstart = affine::makeComposedAffineApply(b, loc, ILstartMap, {IOin, IOout, ILss});

  llvm::errs() << "\niLstartMap:\n";
  ILstartMap.print(llvm::errs());
  llvm::errs() << "\n";

  // HwIndex using ILstart
  AffineExpr iL, k, nwin;
  bindDims(context, iL, k, nwin);
  AffineExpr iHwExpr =
    ((((iL + nwin).floorDiv(Ow) - iL.floorDiv(Ow)) * strideH) +
     ((k % filterHW).floorDiv(Fw))) * Iw +
    (((iL + nwin) % Ow - iL % Ow) * strideW) +
    (k % Fw);

  AffineMap iHwMap = AffineMap::get(3, 0, {iHwExpr}, context);

  llvm::errs() << "\niHwMap:\n";
  iHwMap.print(llvm::errs());
  llvm::errs() << "\n";

  Value HwIndex = affine::makeComposedAffineApply(b, loc, iHwMap, {ILstart, kIndex, nwinIndex});

  return {NcIndex, HwIndex};
}

// Compute the multi-pack input indices considering strides and tensor structure.
static SmallVector<Value, 3> computeMultiPackInputIndices(
    OpBuilder &b, Location loc, Value nIndex, Value tIndex, Value kIndex,
    Value nwinIndex, int64_t Tw, int64_t Fh, int64_t Fw, int64_t Nwin, 
    SmallVector<int64_t, 2> strides) {

  MLIRContext *context = b.getContext();
  AffineExpr iN, iT, iK, iNwin;
  bindDims(context, iN, iT, iK, iNwin);

  //  NcIndex (iNc) = iK floorDiv (Fh * Fw)
  AffineMap NcMap = AffineMap::get(4, 0, {iK.floorDiv(Fh * Fw)}, context);
  Value NcIndex = affine::makeComposedAffineApply(b, loc, NcMap, {nIndex, tIndex, kIndex, nwinIndex});

  // ThIndex (iTh) = ((iT * Nwin * strides[1]) FlooDiv Tw) * strides[0] + (iK mod (Fh * Fw)) FloorDiv Fw
  AffineMap ThMap = AffineMap::get(4, 0, 
          {((iT * Nwin * strides[1]).floorDiv(Tw)) * strides[0] + (iK % (Fh * Fw)).floorDiv(Fw)}, context);
  Value ThIndex = affine::makeComposedAffineApply(b, loc, ThMap, {nIndex, tIndex, kIndex, nwinIndex});

  // TwIndex (iTw) = (iT * Nwin * strides[1]) mod Tw + iNwin * stride[1] + iK mod Fw
  AffineMap TwMap = AffineMap::get(4, 0,
          {(iT * Nwin * strides[1]) % Tw + iNwin * strides[1] + iK % Fw}, context);
  Value TwIndex = affine::makeComposedAffineApply(b, loc, TwMap, {nIndex, tIndex, kIndex, nwinIndex});

  return {NcIndex, ThIndex, TwIndex};
}

// After the packing operations, the linalOps (uKernel) must be fixed to 
// access the packed input & filter. Also, fix the iterator types & maps.
static LogicalResult
adjustLinalgOps(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  // Get context
  MLIRContext *context = rewriter.getContext();

  // Validate input loops
  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Get the body of the loops
  Block *outerBody = outerLoop.getBody();
  Block *innerBody = innerLoop.getBody();

  // Get the uKernel (convOp)
  auto convOp = tiledOps.front();
  Location loc = convOp->getLoc();

  // set the insertion point
  rewriter.setInsertionPoint(convOp);

  // Cast uKernel to linalg::GenericOp
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the uKernel (convOp)");

  // get input & filter based on the schedule
  linalg::GenericOp outPackOp;
  for (Operation &op : outerBody->getOperations()) {
    if (auto packOp = dyn_cast<linalg::GenericOp>(&op)) {
      outPackOp = packOp;
      break;
    }
  }
  linalg::GenericOp innPackOp;
  for (Operation &op : innerBody->getOperations()) {
    if (auto packOp = dyn_cast<linalg::GenericOp>(&op)) {
      innPackOp = packOp;
      break;
    }
  }

  // get the input & filter to new uKernel
  Value input = res.schd == IS ? outPackOp.getResult(0) : innPackOp.getResult(0); 
  Value filter = res.schd == WS ? outPackOp.getResult(0) : innPackOp.getResult(0); 

  // Create the iterator types for new uKernel
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction};

  // Create the new affine maps
  AffineExpr d0, d1, d2, d3;
  bindDims(context, d0, d1, d2, d3);
  auto lhsMap = AffineMap::get(4, 0, {d0, d3, d2}, context);
  auto rhsMap = AffineMap::get(4, 0, {d3, d1}, context);
  auto resultMap = AffineMap::get(4, 0, {d0, d1, d2}, context);

  // create the new uKernel
  auto genericOp = rewriter.create<linalg::GenericOp>(
    loc,
    linalgOp.getOutputs()[0].getType(),
    ValueRange{input, filter},
    linalgOp.getOutputs(),
    ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
    newOpIterators, 
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      Value mul = createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
      Value add = createAdd(loc, mul, args[2], nestedBuilder);
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
    });

  // Replace all uses of the uKernel linalgOp to the new uKernel genericOp
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(genericOp.getResult(0));
  }

  // replace linalgOp with the new uKernel gennericOp
  rewriter.replaceOp(linalgOp, genericOp);

  // Replace all extractOp or affineOp that use the linalgOp.
  outerBody->walk([&](Operation *op) {
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (extractOp.getSource().getDefiningOp() == linalgOp) {
        extractOp.replaceAllUsesWith(genericOp.getResult(0));
      }
    } else if (auto affineOp = dyn_cast<AffineApplyOp>(op)) {
      for (auto &operand : affineOp->getOpOperands()) {
        if (operand.get().getDefiningOp() == linalgOp) {
          operand.set(genericOp.getResult(0));
        }
      }
    }
  });

  // Collect all pending operations and remove them
  SmallVector<Operation *, 4> dependentOps;
  outerBody->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return operand.getDefiningOp() == linalgOp;
        })) {
      dependentOps.push_back(op);
    }
  });
  for (Operation *op : llvm::reverse(dependentOps)) {
    rewriter.eraseOp(op);
  }

  // Iterate through the tiledOps to find convOp and replace it to genericOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = genericOp;
      break;
    }
  }

  return success();
}

// After the inner tile operation, promote the two affine.apply and the first extracted
// slice if chd = IS or the second extracted slice if schd = WS, to the outer loop
static LogicalResult
promoteOpsOfTile(RewriterBase &rewriter, Operation *transformOp,
                 CSAStrategy res, SmallVector<Operation *> loopOps) {

  // Validate input loops
  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Get the body of the loops
  Block *outerBody = outerLoop.getBody();
  Block *innerBody = innerLoop.getBody();

  Operation *affineApply1 = nullptr;
  // Operation *affineApply2 = nullptr;
  Operation *tensorExtractSlice1 = nullptr;
  Operation *tensorExtractSlice2 = nullptr;

  // Locate the target operations within the inner loop body
  for (Operation &op : innerBody->getOperations()) {
    if (!affineApply1 && isa<AffineApplyOp>(&op))
      affineApply1 = &op;
    // else if (!affineApply2 && isa<AffineApplyOp>(&op))
    //   affineApply2 = &op;
    else if (!tensorExtractSlice1 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice1 = &op;
    else if (!tensorExtractSlice2 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice2 = &op;

    // Stop once all target operations are found
    // if (affineApply1 && affineApply2 && tensorExtractSlice1 && tensorExtractSlice2)
    if (affineApply1 && tensorExtractSlice1 && tensorExtractSlice2)
      break;
  }

  // if (!affineApply1 || !affineApply2 || !tensorExtractSlice1 || !tensorExtractSlice2)
  if (!affineApply1 || !tensorExtractSlice1 || !tensorExtractSlice2)
    return transformOp->emitError("Failed to hosting Ops");

  // Set insertion point at the start of outer loop
  rewriter.setInsertionPointToStart(outerBody);

  Operation *promotedAffineApply1 = nullptr;
  // Operation *promotedAffineApply2 = nullptr;
  Operation *promotedTensorExtractSlice1 = nullptr;
  Operation *promotedTensorExtractSlice2 = nullptr;

  // Clone operations to the outer loop
  if (res.schd == IS) {
    SmallVector<Value, 1> outIndVar1 = {outerLoop.getInductionVar()};
    promotedAffineApply1 = rewriter.create<AffineApplyOp>(
      affineApply1->getLoc(), affineApply1->getAttrOfType<AffineMapAttr>("map"), outIndVar1);

    // SmallVector<Value, 1> outIndVar2 = {outerLoop.getInductionVar()};
    // promotedAffineApply2 = rewriter.create<AffineApplyOp>(
    //   affineApply2->getLoc(), affineApply2->getAttrOfType<AffineMapAttr>("map"), outIndVar2);

    promotedTensorExtractSlice1 = rewriter.clone(*tensorExtractSlice1);
  }
  else {
    promotedTensorExtractSlice2 = rewriter.clone(*tensorExtractSlice2);
  }

  // Replace uses of the old operations starting at the outer loop body
  outerBody->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      if (res.schd == IS && operand.get() == affineApply1->getResult(0))
        operand.set(promotedAffineApply1->getResult(0));
      // else if (res.schd == IS && operand.get() == affineApply2->getResult(0))
      //   operand.set(promotedAffineApply2->getResult(0));
      else if (res.schd == IS && operand.get() == tensorExtractSlice1->getResult(0))
        operand.set(promotedTensorExtractSlice1->getResult(0));
      else if (res.schd == WS && operand.get() == tensorExtractSlice2->getResult(0))
        operand.set(promotedTensorExtractSlice2->getResult(0));
    }
  });

  // Erase the old operations
  if (affineApply1->use_empty()) rewriter.eraseOp(affineApply1);
  // if (affineApply2->use_empty()) rewriter.eraseOp(affineApply2);
  if (tensorExtractSlice1->use_empty()) rewriter.eraseOp(tensorExtractSlice1);
  if (tensorExtractSlice2->use_empty()) rewriter.eraseOp(tensorExtractSlice2);

  return success();
}

// Apply the filter packing. This packing will be inserted at the begining of first or
// second loop level of the internal convolution depends of Input or Wheight Stationary
static LogicalResult
applyFilterPacking(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                  SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  // select the loop based on IS or WS
  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  int loopIndex = res.schd == IS ? idx+1 : idx;

  // Cast to scf::ForOp the  selected loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the uKernel
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
applyInputPacking(RewriterBase &rewriter, Operation *transformOp, ConvInfo csaConv,
                  CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides,
                  SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  // select the loops based on IS or WS
  int loopIndex = res.schd == IS ? idx : idx+1;
  int outerIndex = res.schd == IS ? 2 : 3;

  // Cast to scf::ForOp the inner loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the ukernel
  auto convOp = tiledOps.front();

  // Cast to linalg::GenericOp to get the input & filter, types & shapes
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the uKernel");

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
    if (!innerLoop) return transformOp->emitError("failed to get the inner loop");
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
  int64_t iw = csaConv.input_cols; // The iw of the original input
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

  // create the input packing

  Value IO_in  = dyn_cast<scf::ForOp>(loopOps[loopIndex]).getInductionVar();
  Value IO_out = dyn_cast<scf::ForOp>(loopOps[outerIndex]).getInductionVar();
  Value IL_ss = rewriter.create<arith::ConstantIndexOp>(loc, csaConv.split_size);

  auto packingTensor = rewriter.create<linalg::GenericOp>(
    loc, inputPacking.getType(),
    /*inputs=*/ValueRange{}, /*outputs=*/inputPacking, packingIndexingMaps,
    packingIterators,

    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      Value iN = nestedBuilder.create<linalg::IndexOp>(loc, 0);    // n
      Value iK = nestedBuilder.create<linalg::IndexOp>(loc, 1);    // ic * fh * fw
      Value iNwin = nestedBuilder.create<linalg::IndexOp>(loc, 2); // nw

      SmallVector<Value, 2> inputIndices = computeLinearInputIndices(
          nestedBuilder, nestedLoc, iK, iNwin, IO_in, IO_out, IL_ss, fh, fw, ow, iw, strides);
      Value iNc = inputIndices[0];
      Value iHw = inputIndices[1];

      // Create the extraction indices for the original input tensor
      SmallVector<Value> extractionIndices{iN, iNc, iHw};

      // Extract the value from the original input tensor
      Value inputVal = nestedBuilder.create<tensor::ExtractOp>(loc, input, extractionIndices);

      // Yield the extracted value into the packed tensor
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
    });

  return success();
}

// Swap the inner loops when schedule is Input Stationary. This is a workaround.
// Apparently, scf::tileUsingSCF innerInterchange has no effect!
static LogicalResult
swapInductionVars(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                  SmallVector<Operation *> &tiledOps, SmallVector<Operation *> &loopOps) {

  if (res.schd != IS) return success();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Collect induction variables and loop bodies
  Value outerIndVar = outerLoop.getInductionVar();
  Value innerIndVar = innerLoop.getInductionVar();
  Block &outerBody = *outerLoop.getBody();
  Block &innerBody = *innerLoop.getBody();

  // Introduce temporary variable to avoid conflicting replacements
  rewriter.setInsertionPointToStart(&outerBody);
  auto tempOp = rewriter.create<arith::ConstantIndexOp>(outerLoop.getLoc(), 0);
  Value tempIndVar = tempOp.getResult();

  // Replace outerIndVar with a temporary to prevent conflicts
  outerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(outerIndVar, tempIndVar);
  });

  // Replace innerIndVar with outerIndVar
  innerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(innerIndVar, outerIndVar);
  });

  // Replace temporary variable with innerIndVar in the outer loop body
  outerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(tempIndVar, innerIndVar);
  });

  // Erase the temporary variable immediately after the swap
  rewriter.eraseOp(tempOp);

  // Swap loop bounds and steps
  Value outerLowerBound = outerLoop.getLowerBound();
  Value outerUpperBound = outerLoop.getUpperBound();
  Value outerStep = outerLoop.getStep();

  Value innerLowerBound = innerLoop.getLowerBound();
  Value innerUpperBound = innerLoop.getUpperBound();
  Value innerStep = innerLoop.getStep();

  outerLoop.setLowerBound(innerLowerBound);
  outerLoop.setUpperBound(innerUpperBound);
  outerLoop.setStep(innerStep);

  innerLoop.setLowerBound(outerLowerBound);
  innerLoop.setUpperBound(outerUpperBound);
  innerLoop.setStep(outerStep);

  return success();
}

// Pack multiple input tiles, adding another dimension to the packed tensor,
// and skip K * Nwin elements per iteration on WS schedule.
static LogicalResult
inputMultipackingOpt(RewriterBase &rewriter, Operation *transformOp, ConvInfo csaConv,
                     CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides,
                     SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  if (res.schd != WS)
    return success();

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 2 : 3;
  auto nestLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+2]);
  if (!nestLoop || !outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  Location loc = outerLoop.getLoc();
  rewriter.setInsertionPoint(outerLoop);

  // First, find the input & filter extracted_slice in the nestLoop
  Operation *inputSlice = nullptr;
  Operation *filterSlice = nullptr;
  int count = 0;
  nestLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!inputSlice) {  // inputSlice is the first occurrence of ExtractSliceOp
      inputSlice = op;
    }
    if (count == 1) {  // filterSlice is the second occurence
      filterSlice = op;
    }
    count++;
  });

  // Then, get the input extracted_slice used by the old inputPacking
  Operation *extractedSlice = nullptr;
  innerLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!extractedSlice) { // It's the first one in the inner loop
      extractedSlice = op;
    }
  });

  // Last, create the inputMultiPacking with shape {Ni, Ti, K, Nwin} 
  // where Ti = res.k2, K = Nc * Fh * Fw, Nwin = csa.mK_.nwindows
  auto oper0Type = cast<ShapedType>(inputSlice->getOperand(0).getType());
  auto oper0Shape = oper0Type.getShape(); // Used to do a workaround. See comment below

  auto input = inputSlice->getResult(0);
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  auto filter = filterSlice->getResult(0);
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();

  int64_t Ti = res.k2;          // num of tiles to the input 
  int64_t Ni = inputShape[0];
  int64_t Nc = inputShape[1];
  int64_t Nw = inputShape[3];
  int64_t Fh = filterShape[2];
  int64_t Fw = filterShape[3];
  int64_t Nwin = csa.mK_.nwindows;

  // Tw = (colunas do tile do input) - 2 * (Fw / 2)
  if (Nw == oper0Shape[3]-1) Nw++; // This is a workaround in Nw because the scf::tileUsingSCF is reducing
                                   // 1 column from the tile compared to original tensor when stride == 2.
  int64_t Tw = (Nw - 2) * (Fw / 2);

  SmallVector<int64_t, 4> inputPackingShape = {Ni, Ti, Nc * Fh * Fw, Nwin};
  Value inputPacking = rewriter.create<tensor::EmptyOp>(loc, inputPackingShape, inputType.getElementType());

  auto nloops = inputPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 4> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 4> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto newPackingTensor = rewriter.create<linalg::GenericOp>(
    loc, inputPacking.getType(),
    ValueRange{},
    inputPacking,
    packingIndexingMaps,
    packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value iN = nestedBuilder.create<linalg::IndexOp>(loc, 0);
    Value iT = nestedBuilder.create<linalg::IndexOp>(loc, 1);
    Value iK = nestedBuilder.create<linalg::IndexOp>(loc, 2);
    Value iNwin = nestedBuilder.create<linalg::IndexOp>(loc, 3);

    SmallVector<Value, 3> inputIndices = computeMultiPackInputIndices(
        nestedBuilder, nestedLoc, iN, iT, iK, iNwin, Tw, Fh, Fw, Nwin, strides);
    Value iNc = inputIndices[0];
    Value iTh = inputIndices[1];
    Value iTw = inputIndices[2];

    SmallVector<Value> extractionIndices{iN, iNc, iTh, iTw};
    Value inputVal = nestedBuilder.create<tensor::ExtractOp>(loc, inputSlice->getResult(0), extractionIndices);
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
  });

  // Create the affine.apply at beginning of innerLoop body
  // to indexing the new inputSlice
  rewriter.setInsertionPointToStart(innerLoop.getBody());
  Value affineIndex = rewriter.create<AffineApplyOp>(
      innerLoop.getLoc(), AffineMap::get(1, 0, getAffineDimExpr(0, context).floorDiv(Nwin), context), innerLoop.getInductionVar());
 
  // Define new offsets, sizes, and strides for new extractedSlice
  SmallVector<OpFoldResult, 4> newSliceOffsets = {
      rewriter.getIndexAttr(0), affineIndex, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0) };
  SmallVector<OpFoldResult, 4> newSliceSizes = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(inputPackingShape[2]), rewriter.getIndexAttr(inputPackingShape[3]) };
  SmallVector<OpFoldResult, 4> newSliceStrides = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };

  Value updatedExtractedSlice = rewriter.create<tensor::ExtractSliceOp>(
    innerLoop.getLoc(), newPackingTensor.getResult(0), newSliceOffsets, newSliceSizes, newSliceStrides);

  // Apply tensor.collapse_shape on the first two dimensions of new extractedSlice
  SmallVector<ReassociationIndices> collapseDims = {{0, 1}, {2}, {3}};
  Value collapsedSlice = rewriter.create<tensor::CollapseShapeOp>(
      innerLoop.getLoc(),
      RankedTensorType::get({inputPackingShape[0], inputPackingShape[2], inputPackingShape[3]}, inputType.getElementType()),
      updatedExtractedSlice,
      collapseDims);

  // Get the old uKernel & old inputPacking
  linalg::GenericOp linalgOp;
  linalg::GenericOp oInputPacking;
  innerLoop.getBody()->walk([&](linalg::GenericOp op) {
    if (!oInputPacking) {
      oInputPacking = op; // Must be the first one
    }
    linalgOp = op; // It's the last one in the loop
  });

  // Set the insertion point to old uKernel
  rewriter.setInsertionPoint(linalgOp);

  auto linalgInputs = linalgOp.getInputs(); 
  SmallVector<Value, 2> newInputs;
  newInputs.push_back(collapsedSlice);  // Replace old inputPacking
  newInputs.push_back(linalgInputs[1]); // Same filterPacking

  // Get the indexingMaps of linalgOp
  SmallVector<AffineMap, 3> indexingMaps;
  if (auto indexingMapsAttr = linalgOp.getIndexingMaps()) {
    for (auto attr : indexingMapsAttr) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
        indexingMaps.push_back(affineMapAttr.getValue());
      } else {
        return transformOp->emitError("Expected AffineMapAttr in indexingMaps.");
      }
    }
  } else {
    return transformOp->emitError("IndexingMaps not found on linalgOp.");
  }

  // Create the iterator types for new uKernel
  auto reduction = utils::IteratorType::reduction; // parallel already defined
  SmallVector<utils::IteratorType> iteratorTypes = {parallel, parallel, parallel, reduction};

  // create the new uKernel
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
    linalgOp.getLoc(), linalgOp->getResultTypes(), 
    newInputs, linalgOp.getOutputs(),
    indexingMaps, iteratorTypes);

  // clone the body of linalgOp into the new uKernel
  rewriter.inlineRegionBefore(linalgOp.getRegion(), newLinalgOp.getRegion(),
                              newLinalgOp.getRegion().begin());

  // Replace all uses of the uKernel linalgOp to new uKernel
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(newLinalgOp.getResult(0));
  }

  // replace linalgOp with the new uKernel
  rewriter.replaceOp(linalgOp, newLinalgOp->getResults());

  Operation *emptyOp = nullptr;
  innerLoop.getBody()->walk([&](tensor::EmptyOp op) {
    emptyOp = op;
  });

  // Locate the old affineApply operations within the inner loop body
  Operation *affineApply1 = nullptr;
  Operation *affineApply2 = nullptr;
  Operation *affineApply3 = nullptr;
  for (Operation &op : innerLoop.getBody()->getOperations()) {
    if (!affineApply1 && isa<AffineApplyOp>(&op))
      affineApply1 = &op; // It's the new AffineApply inserted 
    else if (!affineApply2 && isa<AffineApplyOp>(&op))
      affineApply2 = &op;
    else if (!affineApply3 && isa<AffineApplyOp>(&op))
      affineApply3 = &op;
    // Stop once all target operations are found
    if (affineApply1 && affineApply2 && affineApply3)
      break;
  }

  // Remove obsolet ops
  SmallVector<Operation *, 4> opsToRemove;
  opsToRemove.push_back(emptyOp);
  opsToRemove.push_back(affineApply2);
  opsToRemove.push_back(affineApply3);
  opsToRemove.push_back(extractedSlice); 
  opsToRemove.push_back(oInputPacking);

  for (Operation *op : llvm::reverse(opsToRemove)) {
    if (op && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  auto convOp = tiledOps.front();
  // Iterate through the tiledOps to find convOp and replace it to newLinalgOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = newLinalgOp;
      break;
    }
  }

  return success();

}

// Pack multiple filter tiles adding a new dimension to the tensor k2 which
// iterates over groups of Nf filters on IS.
static LogicalResult
filterMultipackingOpt(RewriterBase &rewriter, Operation *transformOp,
                      CSAStrategy res, SmallVector<Operation *> &tiledOps,
                      SmallVector<Operation *> loopOps) {
  if (res.schd != IS)
    return success();

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 2 : 3;
  auto nestLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+2]);
  if (!nestLoop || !outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  Location loc = outerLoop.getLoc();
  rewriter.setInsertionPoint(outerLoop);

  // First, find the filter extracted_slice in the nestLoop
  Value filterSlice;
  int count = 0;
  nestLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (count == 1) {  // It's the second occurrence of ExtractSliceOp
      filterSlice = op;
    }
    count++;
  });

  // Then, get the filter extracted_slice used by the old filterPacking
  Operation *extractedSlice = nullptr;
  innerLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!extractedSlice) { // It's the first one in the inner loop
      extractedSlice = op;
    }
  });

  // Last, create the filterMultiPacking with shape {Tf, Nc × Fh × Fw, Nf}
  auto filter = extractedSlice->getResult(0);
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();
  int64_t Tf = res.k2;          // num of tiles to the filter 
  int64_t Nf = filterShape[0];
  int64_t Nc = filterShape[1];
  int64_t Fh = filterShape[2];
  int64_t Fw = filterShape[3];
  SmallVector<int64_t, 3> filterPackingShape = {Tf, Nc * Fh * Fw, Nf};
  Value filterPacking = rewriter.create<tensor::EmptyOp>(loc, filterPackingShape, filterType.getElementType());

  auto nloops = filterPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 3> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 3> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto newPackingTensor = rewriter.create<linalg::GenericOp>(
    loc, filterPacking.getType(),
    ValueRange{},
    filterPacking,
    packingIndexingMaps,
    packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value index0 = nestedBuilder.create<linalg::IndexOp>(loc, 0);
    Value index1 = nestedBuilder.create<linalg::IndexOp>(loc, 1);
    Value index2 = nestedBuilder.create<linalg::IndexOp>(loc, 2);

    SmallVector<Value> unpackedIndices = unrollIndex(
      nestedBuilder, nestedLoc, index1, ArrayRef<int64_t>{Nc, Fh, Fw});
    Value NcIndex = unpackedIndices[0];
    Value FhIndex = unpackedIndices[1];
    Value FwIndex = unpackedIndices[2];

    Value kIndex = computeFilterIndices(nestedBuilder, nestedLoc, index0, index2, Nf);
    SmallVector<Value> extractionIndices{kIndex, NcIndex, FhIndex, FwIndex};

    Value filterVal = nestedBuilder.create<tensor::ExtractOp>(loc, filterSlice, extractionIndices);
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, filterVal);
  });

  // Create the affine.apply at beginning of innerLoop body
  // to indexing the new filterSlice
  rewriter.setInsertionPointToStart(innerLoop.getBody());
  Value affineIndex = rewriter.create<AffineApplyOp>(
      innerLoop.getLoc(), AffineMap::get(1, 0, getAffineDimExpr(0, context).floorDiv(Nf), context), innerLoop.getInductionVar());

  // Define new offsets, sizes, and strides for new extractedSlice
  SmallVector<OpFoldResult, 3> newSliceOffsets = {
      affineIndex, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0) };
  SmallVector<OpFoldResult, 3> newSliceSizes = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(filterPackingShape[1]), rewriter.getIndexAttr(filterPackingShape[2]) };
  SmallVector<OpFoldResult, 3> newSliceStrides = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };

  Value updatedExtractedSlice = rewriter.create<tensor::ExtractSliceOp>(
    innerLoop.getLoc(), newPackingTensor.getResult(0), newSliceOffsets, newSliceSizes, newSliceStrides);

  // Apply tensor.collapse_shape on the first two dimensions of new extractedSlice
  SmallVector<ReassociationIndices> collapseDims = {{0, 1}, {2}};
  Value collapsedSlice = rewriter.create<tensor::CollapseShapeOp>(
      innerLoop.getLoc(),
      RankedTensorType::get({filterPackingShape[1], filterPackingShape[2]}, filterType.getElementType()),
      updatedExtractedSlice,
      collapseDims);

  // Get the old uKernel & old filterPacking
  linalg::GenericOp linalgOp;
  linalg::GenericOp oFilterPacking;
  innerLoop.getBody()->walk([&](linalg::GenericOp op) {
    if (!oFilterPacking) {
      oFilterPacking = op; // Must be the first one
    }
    linalgOp = op; // It's the last one in the loop
  });

  // Set the insertion point to old uKernel
  rewriter.setInsertionPoint(linalgOp);

  auto linalgInputs = linalgOp.getInputs(); 
  SmallVector<Value, 2> newInputs;
  newInputs.push_back(linalgInputs[0]); // Same inputPacking
  newInputs.push_back(collapsedSlice);  // Replace old filterPacking

  // Get the indexingMaps of linalgOp
  SmallVector<AffineMap, 3> indexingMaps;
  if (auto indexingMapsAttr = linalgOp.getIndexingMaps()) {
    for (auto attr : indexingMapsAttr) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
        indexingMaps.push_back(affineMapAttr.getValue());
      } else {
        return transformOp->emitError("Expected AffineMapAttr in indexingMaps.");
      }
    }
  } else {
    return transformOp->emitError("IndexingMaps not found on linalgOp.");
  }

  // Create the iterator types for new uKernel
  auto reduction = utils::IteratorType::reduction; // parallel already defined
  SmallVector<utils::IteratorType> iteratorTypes = {parallel, parallel, parallel, reduction};

  // create the new uKernel
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
    linalgOp.getLoc(), linalgOp->getResultTypes(), 
    newInputs, linalgOp.getOutputs(),
    indexingMaps, iteratorTypes);

  // clone the body of linalgOp into the new uKernel
  rewriter.inlineRegionBefore(linalgOp.getRegion(), newLinalgOp.getRegion(),
                              newLinalgOp.getRegion().begin());

  // Replace all uses of the uKernel linalgOp to new uKernel
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(newLinalgOp.getResult(0));
  }

  // replace linalgOp with the new uKernel
  rewriter.replaceOp(linalgOp, newLinalgOp->getResults());

  Operation *emptyOp = nullptr;
  innerLoop.getBody()->walk([&](tensor::EmptyOp op) {
    emptyOp = op;
  });

  // Remove obsolet ops
  SmallVector<Operation *, 4> opsToRemove;
  opsToRemove.push_back(emptyOp);
  opsToRemove.push_back(extractedSlice); 
  opsToRemove.push_back(oFilterPacking);

  for (Operation *op : llvm::reverse(opsToRemove)) {
    if (op && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  auto convOp = tiledOps.front();
  // Iterate through the tiledOps to find convOp and replace it to newLinalgOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = newLinalgOp;
      break;
    }
  }

  return success();

}

// Apply a tiling transformation to a modified payload ops and store both the
// tiled operation  (uKernel) as well as the created tile loops.
static LogicalResult
applyTileTo(RewriterBase &rewriter, Operation *transformOp, Operation *target,
            ConvInfo csaConv, CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, 
            SmallVector<Operation*, 7> &outResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Assign tile sizes
  // Input Stationary: N, Nf * K2, Nwin * K3, Nc, Fh, Fw  
  // Weight Stationary: N, Nf * K3, Nwin * K2, Nc, Fh, Fw  
  int64_t nFTiles = csa.mK_.num_filters * (res.schd == IS ? res.k2 : res.k3);
  int64_t nWinTiles = csa.mK_.nwindows * (res.schd == IS ? res.k3 : res.k2);
  SmallVector<int64_t, 6> tileSize = {1, nFTiles, nWinTiles, res.tile_c, 0, 0};
  SmallVector<OpFoldResult> tileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), tileSize);

  // Order:
  // Input Stationary: N, Nc, Nwin, Nf
  // Weight Stationary: N, Nc, Nf, Nwin
  int64_t outer = res.schd == IS ? 2 : 1;
  int64_t inner = res.schd == IS ? 1 : 2;
  SmallVector<int64_t, 4> tileInterchange = {0, 3, outer, inner};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOfr).setInterchange(tileInterchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);
  if (failed(tiledResults))
    return transformOp->emitError("Outermost tiling operation was failed ");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(tilingInterfaceOp, tiledResults->loops.empty()
                             ? tiledResults->tiledOps.front()->getResults()
                             : tiledResults->loops.front()->getResults());

  // Perform the tiling in the inner convolution
  auto innerOp = tiledResults->tiledOps.front();

  SmallVector<int64_t, 6> innerTileSize = {0, csa.mK_.num_filters, csa.mK_.nwindows, 0, 0, 0};
  SmallVector<OpFoldResult> innerTileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), innerTileSize);

  int64_t innerFOrder = res.schd == IS ? 1 : 0;
  int64_t innerSOrder = res.schd == IS ? 0 : 1;
  SmallVector<int64_t, 2> innerInterchange = {innerFOrder, innerSOrder};

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
    return transformOp->emitError("Innermost tiling operation was failed ");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(innerTilingInterfaceOp, innerTiledResults->loops.empty()
                             ? innerTiledResults->tiledOps.front()->getResults()
                             : innerTiledResults->loops.front()->getResults());

  // Report back the relevant handles to the transform op.
  tiledOps.push_back(innerTiledResults->tiledOps.front());
  for (Operation *loop : tiledResults->loops)
    loopOps.push_back(loop);
  for (Operation *loop : innerTiledResults->loops)
    loopOps.push_back(loop);

  // For debug only:
  auto rootLoop = dyn_cast<scf::ForOp>(loopOps[0]);
  llvm::errs() << "\n=== Loops after tiling === \n" << rootLoop << "\n\n";

  // Swap the innner loops in the case of Input Stationary
  LogicalResult result0 = swapInductionVars(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result0)) return transformOp->emitError("failed to swap indvar Ops");

  // Promote some inner loop Ops depending on the schedule (WS or IS)
  LogicalResult result1 = promoteOpsOfTile(rewriter, transformOp, res, loopOps);
  if (failed(result1)) return transformOp->emitError("failed to hosting Ops");

  // // Generate the filter packing
  LogicalResult result2 = applyFilterPacking(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result2)) return transformOp->emitError("failed to apply the filter packing");

  // Generate the input packing
  LogicalResult result3 = applyInputPacking(rewriter, transformOp, csaConv, csa, res, strides, tiledOps, loopOps);
  if (failed(result3)) return transformOp->emitError("failed to apply the input packing");

  // Fix the uKernel after packing input & filter
  LogicalResult result4 = adjustLinalgOps(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result4)) return transformOp->emitError("failed to replace the uKernel after packing");

  // // Generate the filter Multi-Packing
  // LogicalResult result5 = filterMultipackingOpt(rewriter, transformOp, res, tiledOps, loopOps);
  // if (failed(result5)) return transformOp->emitError("failed to apply filter Multi-Packing optimization");

  // // Generate the input Multi-Packing
  // LogicalResult result6 = inputMultipackingOpt(rewriter, transformOp, csaConv, csa, res, strides, tiledOps, loopOps);
  // if (failed(result6)) return transformOp->emitError("failed to apply input Multi-Packing optimization");

  // Store the results (Operation*) in the output variable (as Value)
  outResults.push_back(tiledOps.front());  // The head is the linalg.generic (uKernel)
  for (auto &loop : loopOps)
    outResults.push_back(loop);  // The tail contain the loops

  return success();
}

/// Validates the inputs for linalg::splitOp.
/// Ensures that the dimension is valid and that the splitPoint, offsets, and sizes
/// are well-formed (i.e., static and within bounds).
LogicalResult validateSplitInputs(RewriterBase &rewriter, Operation *transformOp,
                                  TilingInterface op, unsigned dim,
                                  OpFoldResult splitPoint) {

  Location loc = op->getLoc();
  auto iterationSpace = op.getIterationDomain(rewriter);

  if (dim >= iterationSpace.size()) {
    return transformOp->emitError()
           << "Split dimension " << dim << " exceeds iteration domain size "
           << iterationSpace.size() << ".";
  }

  auto offset = iterationSpace[dim].offset;
  auto size = iterationSpace[dim].size;

  // for debug only:
  llvm::errs() << "=== Iteration Domain Sizes ===\n";
  for (unsigned i = 0; i < iterationSpace.size(); ++i) {
    llvm::errs() << "Dim " << i << ": ";
    OpFoldResult size = iterationSpace[i].size;
    if (auto attr = size.dyn_cast<Attribute>()) {
      attr.print(llvm::errs());
    } else if (auto val = size.dyn_cast<Value>()) {
      val.print(llvm::errs());
    } else {
      llvm::errs() << "Unknown\n";
    }
    llvm::errs() << "\n";
  }

  // Ensure splitPoint is an index attribute (static)
  auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
  if (!splitAttr)
    return transformOp->emitError("Expected static split point.");

  auto offsetAttr = llvm::dyn_cast_if_present<Attribute>(offset);
  auto sizeAttr = llvm::dyn_cast_if_present<Attribute>(size);
  if (!offsetAttr || !sizeAttr)
    return transformOp->emitError("Expected static offset and size in iteration space.");

  int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
  int64_t offsetVal = cast<IntegerAttr>(offsetAttr).getInt();
  int64_t sizeVal = cast<IntegerAttr>(sizeAttr).getInt();

  if (splitVal <= 0 || splitVal >= sizeVal) {
    return transformOp->emitError()
           << "Invalid split point " << splitVal
           << " for iteration space size " << sizeVal << ".";
  }

  return success();
}

/// Splits a linalg.generic op along a given dimension and splitPoint.
/// This assumes the dimension is splittable (CSA already validated bounds).
std::pair<TilingInterface, TilingInterface>
performSplit(RewriterBase &rewriter, TilingInterface op,
             unsigned dimension, OpFoldResult splitPoint) {

  Location loc = op->getLoc();
  MLIRContext *context = rewriter.getContext();

  rewriter.setInsertionPoint(op); 

  // Defensive check: ensure dimension is valid
  auto iterationSpace = op.getIterationDomain(rewriter);
  if (dimension >= iterationSpace.size()) {
    llvm::errs() << "split dimension " << dimension << " out of bounds\n";
    return {TilingInterface(), TilingInterface()};
  }

  // Defensive check: static splitPoint must not exceed the size
  if (auto sizeAttr = iterationSpace[dimension].size.dyn_cast<Attribute>()) {
    auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
    if (splitAttr) {
      int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
      int64_t dimSize = cast<IntegerAttr>(sizeAttr).getInt();
      if (splitVal <= 0 || splitVal >= dimSize) {
        llvm::errs() << "splitPoint " << splitVal << " is invalid for dimension size " << dimSize << "\n";
        return {TilingInterface(), TilingInterface()};
      }
    }
  }

  // Perform the actual split
  return linalg::splitOp(rewriter, op, dimension, splitPoint);
}

/// This function will be called when the convolution needs to be Splitted.
/// It takes the original convolution (`genericOp`), performs the split, and then applies tiling.
static LogicalResult
splitAndTileConvolution(RewriterBase &rewriter, Operation *transformOp, Operation* target,
                        ConvInfo csaConv, CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides,
                        SmallVector<SmallVector<Operation*, 6>> &resultLoops,
                        SmallVector<Operation*> &resultConvs) {

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  MLIRContext *context = rewriter.getContext();
  Location loc = target->getLoc();

  // Define the dimension & split point to split the convolution
  // For now, supported split in one dimension only
  CSAStrategy split_res = res;

  int64_t splitDim;
  int64_t splitSize;
  int64_t extraSize;

  if (res.extra_tile_c !=0) {
    splitDim = 3;
    splitSize = csaConv.input_channels - res.extra_tile_c;
    split_res.tile_c = 0;
  }
  else if (res.extra_k2 != 0) {
    splitDim = res.schd == WS ? 2 : 1;
    extraSize = res.schd == WS ? csa.mK_.nwindows * res.extra_k2 : csa.mK_.num_filters * res.extra_k2;
    splitSize = csaConv.output_rows * csaConv.output_cols - extraSize;
    split_res.k2 = res.extra_k2;
  }
  else if (res.extra_k3 != 0) {
    splitDim = res.schd == IS ? 2 : 1;
    extraSize = res.schd == IS ? csa.mK_.nwindows * res.extra_k3 : csa.mK_.num_filters * res.extra_k3;
    splitSize = csaConv.output_rows * csaConv.output_cols - extraSize;
    split_res.k3 = res.extra_k3;
  }

  OpFoldResult splitPoint = rewriter.getIndexAttr(splitSize);
  if (failed(validateSplitInputs(rewriter, transformOp, tilingInterfaceOp, splitDim, splitPoint))) {
    return transformOp->emitError("Invalid dimension or splitPoint to call linalg::splitOp"); 
  }
  auto [firstOp, secondOp] = performSplit(rewriter, tilingInterfaceOp, splitDim, splitPoint);

  // For debug only:
  llvm::errs() << "=== Splitted kernels ===\n";
  llvm::errs() << "First :\n ";
  firstOp.print(llvm::errs());
  llvm::errs() << "\nLast :\n ";
  secondOp.print(llvm::errs());

  // Apply the tiling for each part of the split
  SmallVector<Operation*, 7> firstResults, secondResults;
  
  rewriter.setInsertionPoint(target);

  // In the prologue split, csaConv.split_size equals 0
  csaConv.split_size = 0;
  if (failed(applyTileTo(rewriter, transformOp, firstOp, csaConv, csa, res, strides, firstResults))) {
    return transformOp->emitError("Failed to apply tiling on first convOp after split.");
  }
  resultConvs.push_back(firstResults[0]);
  SmallVector<Operation*, 6> firstLoopSet;
  for (int i = 1; i <= 6; ++i) {
    firstLoopSet.push_back(firstResults[i]);
  }
  resultLoops.push_back(firstLoopSet);

  // For debug only:
  // auto root1Loop = dyn_cast<scf::ForOp>(firstResults[1]);
  // llvm::errs() << "Loops after tiling for first convOp: \n" << root1Loop << "\n\n";

  // In the epilogue split, csaConv.split_size equals splitSize
  csaConv.split_size = splitSize;
  if (failed(applyTileTo(rewriter, transformOp, secondOp, csaConv, csa, split_res, strides, secondResults))) {
    return transformOp->emitError("Failed to apply tiling on second convOp after split.");
  }
  resultConvs.push_back(secondResults[0]);
  SmallVector<Operation*, 6> secondLoopSet;
  for (int i = 1; i < secondResults.size(); ++i) {
    secondLoopSet.push_back(secondResults[i]);
  }
  resultLoops.push_back(secondLoopSet);

  // For debug only:
  // auto root2Loop = dyn_cast<scf::ForOp>(secondResults[1]);
  // llvm::errs() << "Loops after tiling for second convOp: \n" << root2Loop << "\n\n";

  return success();
}

///
/// Implementation of SConv::apply transform dialect operation.
///
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Initialize the default values of mKInfo & ArchInfo to the CSA Analysis.
  // It's dependent of the target machine. We used these values for Sorgan
  mKInfo mK = {16, 8, 128};
  ArchInfo arch = {
      (uint32_t)(32768 * 0.9),
      (uint32_t)(1048576 * 0.9),
      (uint32_t)(4194304 * 0.9),
      2, 10, 30, 300, 128
  };

  // Get the optional arguments
  auto mKInfoAttr = getMKInfo();
  auto archInfoAttr = getArchInfo();
  auto latencyAttr = getLatency();

  // If `mKInfoAttr` was provided, use the given values
  if (mKInfoAttr) {
    SmallVector<int64_t, 4> mKValues;
    for (auto attr : mKInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        mKValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: mKInfoAttr contains non-integer values!\n";
      }
    }
    if (mKValues.size() >= 2) {
      mK.nwindows = mKValues[0];
      mK.num_filters = mKValues[1];
      mK.noutput = mK.nwindows * mK.num_filters;
    } else {
      return emitSilenceableError() << "Error: mKInfoAttr does not contain enough values!\n";
    }
  }

  // If `archInfoAttr` was provided, use the given values
  if (archInfoAttr) {
    SmallVector<int64_t, 4> archValues;
    for (auto attr : archInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        archValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: archInfoAttr contains non-integer values!\n";
      }
    }
    if (archValues.size() >=4) {
      arch.l1_size = (uint32_t)(archValues[0] * 0.9);
      arch.l2_size = (uint32_t)(archValues[1] * 0.9);
      arch.l3_size = (uint32_t)(archValues[2] * 0.9);
      arch.cache_line = archValues[3];
    } else {
      return emitSilenceableError() << "Error: archInfoAttr does not contain enough values!\n";
    }
  }

  // If `latencyAttr` was provided, use the given values
  if (latencyAttr) {
    SmallVector<int64_t, 4> latencyValues;
    for (auto attr : latencyAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        latencyValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: latencyAttr contains non-integer values!\n";
      }
    }
    if (latencyValues.size() >=4) {
      arch.l1_latency = latencyValues[0];
      arch.l2_latency = latencyValues[1];
      arch.l3_latency = latencyValues[2];
      arch.mem_latency = latencyValues[3];
    } else {
      return emitSilenceableError() << "Error: latencyAttr does not contain enough values!\n";
    }
  }

  // temporary variables to store all conv transformation
  SmallVector<Operation*> tempResultConvs;
  SmallVector<SmallVector<Operation*, 6>> tempResultLoops;

  // Get context and convOps
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  for (Operation *targetOp : targetOps) {
    auto convOp = dyn_cast_or_null<linalg::Conv2DNchwFchwOp>(targetOp);
    if (!convOp)
      continue;  // expected only Conv2DNchwFchw convolutions for transformations

    // Starting generalize the named convolution
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

    // SConv do not support dilation, for now.
    if (!hasAllOneValues(convOp.getDilations()))
      return emitSilenceableError() << "expected all ones for dilations";

    auto inputShape = inputType.getShape();
    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();
    int64_t n = outputShape[0];
    int64_t ic = inputShape[1];
    int64_t ih = inputShape[2];
    int64_t iw = inputShape[3];
    int64_t fh = filterShape[2];
    int64_t fw = filterShape[3];
    int64_t oc = outputShape[1];
    int64_t oh = outputShape[2];
    int64_t ow = outputShape[3];

    // Create the Collapsed shape of input tensor
    SmallVector<ReassociationIndices> inputReassocIndices = {{0}, {1}, {2, 3}};
    auto reshapedInputType = RankedTensorType::get({n, ic, ih * iw}, inputType.getElementType());
    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, inputReassocIndices);

    // Create the Collapsed shape of output tensor
    SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
    auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    // Create the affine maps, iterator types and output tensor shape
    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

    // Get the strides
    auto hstride = convOp.getStrides().getValues<int64_t>()[0];
    auto wstride = convOp.getStrides().getValues<int64_t>()[1];
    SmallVector<int64_t, 2> strides = {hstride, wstride};

    // Affine Expr definitions:
    // d0 = batch; d1 = filter; d2 = output height; d3 = output width; d4 = channels; d5 = filter height; d6 = filter width

    AffineExpr d0, d1, d3, d4, d5, d6;
    bindDims(context, d0, d1, d3, d4, d5, d6);
    auto lhsMap = AffineMap::get(6, 0, {d0, d4, (d3.floorDiv(oh) * hstride + d5) * ih + d3 % oh * wstride + d6}, context);
    auto rhsMap = AffineMap::get(6, 0, {d1, d4, d5, d6}, context);
    auto resultMap = AffineMap::get(6, 0, {d0, d1, d3}, context);

    // Create the new genericOp that replaces the named convolution
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        reshapedOutputType,
        ValueRange{reshapedInput, inputs[1]},
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
    ConvInfo csaConv = {ic, iw, oh, ow, fh, fw, oc, 0, 4};
    CSA csa = createCSAPass(arch, csaConv, mK);
    CSAStrategy res = csa();
    
    // Apply the split (if necessary) & tiling in the genericOp based on the CSA Analysis
    bool requiresSplit = (res.extra_k2 != 0) || (res.extra_k3 != 0) || (res.extra_tile_c != 0);
    if (requiresSplit) {
      if (failed(splitAndTileConvolution(rewriter, getOperation(), genericOp, csaConv, csa, res, strides, tempResultLoops, tempResultConvs)))
        return emitSilenceableError() << "Failed to apply split & tiling to the convolution operation";
    }
    else {
      SmallVector<Operation*, 7> localResults;
      // If no split then csaConv.split_size equals 0
      csaConv.split_size = 0;
      if (failed(applyTileTo(rewriter, getOperation(), genericOp, csaConv, csa, res, strides, localResults)))
        return emitSilenceableError() << "Failed to apply tiling to one of the convolution operations";
      // The first result is the transformed linalg.generic (uKernel)
      tempResultConvs.push_back(localResults[0]);
      // Following, the generated loops
      SmallVector<Operation*, 6> loopSet;
      for (int i = 1; i <= 6; ++i)
        loopSet.push_back(localResults[i]);
      tempResultLoops.push_back(loopSet);
    }
  }

  // Flatten tempResultLoops
  SmallVector<Operation*> flatResultLoops;
  for (const auto &loopSet : tempResultLoops) {
    flatResultLoops.append(loopSet.begin(), loopSet.end());
  }

  // Store results properly in TransformResults
  results.set(getOperation()->getOpResult(0), tempResultConvs);
  results.set(getOperation()->getOpResult(1), flatResultLoops);

  return DiagnosedSilenceableFailure::success();
}

void transform::SConvOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult
transform::SConvOp::verify() {
  // All necessary checks are done in the Apply
  return success();
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
