//===-- SplitToShards.cpp - Split the micro kernels into Shards -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define the shards to NPU parallel operation of the SConv transform
// dialect extension.
//
//===----------------------------------------------------------------------===//
#include <cassert>

#include "SConv.h"
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>
#include <cstdint>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// Split the uKernel into shards to NPU parallel operation
//===----------------------------------------------------------------------===//

namespace {

/// Validates the inputs for linalg::splitOp.
/// Ensures that the dimension is valid and that the splitPoint, offsets, and sizes
/// are well-formed (i.e., static and within bounds).
LogicalResult validateSplit(RewriterBase &rewriter, Operation *transformOp,
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

  LLVM_DEBUG({ 
    DBGS() << "=== Iteration Space ===";
    for (unsigned i = 0; i < iterationSpace.size(); ++i) {
      DBGS() << "Dim " << i << ": ";
      OpFoldResult size = iterationSpace[i].size;
      if (auto attr = size.dyn_cast<Attribute>()) {
        attr.print(llvm::dbgs());
      } else if (auto val = size.dyn_cast<Value>()) {
        val.print(llvm::dbgs());
      } else {
        DBGS() << "Unknown\n";
      }
    }
  });

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

  if (splitVal >= sizeVal) {
    return transformOp->emitError()
           << "Invalid split point " << splitVal
           << " for iteration space size " << sizeVal << ".";
  }

  LLVM_DEBUG({ 
    DBGS() << "=== Split Point: Dim " << dim << " Size " << splitVal;
  });

  return success();
}

/// Splits a linalg.generic op along a given dimension and splitPoint.
/// This assumes the dimension is splittable.
std::pair<TilingInterface, TilingInterface>
computeSplit(RewriterBase &rewriter, TilingInterface op,
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

  // It actually performs the split
  return linalg::splitOp(rewriter, op, dimension, splitPoint);
}


static LogicalResult splitKernel(
    RewriterBase &rewriter, Operation *shardOp, Operation *target, int64_t splitDim,
    int64_t splitSize, Operation *&leftOp, Operation *&rightOp) {

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return shardOp->emitError("only TilingInterface ops are supported to split");

  OpFoldResult splitPoint = rewriter.getIndexAttr(splitSize);
  if (failed(validateSplit(rewriter, shardOp, tilingInterfaceOp, splitDim, splitPoint))) {
    return shardOp->emitError("Invalid dimension or splitPoint to call linalg::splitOp");
  }

  auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
  int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
  std::tie(leftOp, rightOp) = computeSplit(rewriter, tilingInterfaceOp, splitDim, splitPoint);

  LLVM_DEBUG({
    DBGS() << "=== Splitted kernel with Dim = " << splitDim << ", Split Size = " << splitSize << " ===\n";
    DBGS() << "Left :\n";
    leftOp->print(llvm::dbgs());
    DBGS() << "Right :\n";
    rightOp->print(llvm::dbgs());
  });

  return success();
}

static FailureOr<Operation *> splitKernelIntoShards(TransformRewriter &rewriter,
                                                    SplitToShardsOp shardOp,
                                                    Operation *kernelOp) {

  // split kernelOp along "Oc" into 4 shards.
  auto schedAttr = kernelOp->getAttrOfType<mlir::StringAttr>("schedule");
  assert(schedAttr && "expected 'schedule' string attr on ukernel");
  llvm::StringRef sched = schedAttr.getValue();
  int64_t splitDim = (sched == "IS") ? 3 : 2;

  auto op = dyn_cast<TilingInterface>(kernelOp);
  auto iterationSpace = op.getIterationDomain(rewriter);
  auto sizeAttr = iterationSpace[splitDim].size.dyn_cast<Attribute>();
  int64_t dimSize = cast<IntegerAttr>(sizeAttr).getInt();
  int64_t splitSize = llvm::divideCeil(dimSize, 4);

  Operation *firstOp = nullptr, *secondOp = nullptr;
  Operation *thirdOp = nullptr, *forthOp = nullptr;
  Operation *restOp = nullptr;

  if (failed(splitKernel(rewriter, shardOp, kernelOp, splitDim, splitSize, firstOp, restOp)))
    return shardOp->emitError("Failed to split uKernel into shards.");
  if (failed(splitKernel(rewriter, shardOp, restOp, splitDim, splitSize, secondOp, restOp)))
    return shardOp->emitError("Failed to split uKernel into shards.");
  if (failed(splitKernel(rewriter, shardOp, restOp, splitDim, splitSize, thirdOp, forthOp)))
    return shardOp->emitError("Failed to split uKernel into shards.");

  auto schedule = kernelOp->getAttr("schedule");
  firstOp->setAttrs({{"microkernel", rewriter.getUnitAttr()},
                     {"schedule", schedule},
                     {"npu.id", rewriter.getIndexAttr((int64_t)0)}});
  secondOp->setAttrs({{"microkernel", rewriter.getUnitAttr()},
                     {"schedule", schedule},
                     {"npu.id", rewriter.getIndexAttr((int64_t)1)}});
  thirdOp->setAttrs({{"microkernel", rewriter.getUnitAttr()},
                     {"schedule", schedule},
                     {"npu.id", rewriter.getIndexAttr((int64_t)2)}});
  forthOp->setAttrs({{"microkernel", rewriter.getUnitAttr()},
                     {"schedule", schedule},
                     {"npu.id", rewriter.getIndexAttr((int64_t)3)}});

  return forthOp;
}

} // namespace
DiagnosedSilenceableFailure SplitToShardsOp::apply(TransformRewriter &rewriter,
                                                   TransformResults &results,
                                                   TransformState &state) {
  SmallVector<Operation *, 6> resultOps;
  auto targetOps = state.getPayloadOps(getTarget());

  for (Operation *targetOp : targetOps) {
    auto ukernel = dyn_cast<GenericOp>(targetOp);
    if (!ukernel) {
      ukernel.emitError() << "expected `linalg.generic` operation";
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    if (!ukernel->hasAttr("microkernel")) {
      ukernel.emitWarning()
          << "ignoring operation without `microkernel` attribute";
      continue;
    }

    auto result = splitKernelIntoShards(rewriter, *this, ukernel);
    if (failed(result))
      return DiagnosedSilenceableFailure::definiteFailure();

    resultOps.push_back(*result);
  }

  results.set((*this)->getOpResult(0), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void SplitToShardsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult SplitToShardsOp::verify() {
  return success();
}
