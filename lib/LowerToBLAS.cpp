//===-- Lowering.cpp - Lower micro kernels to BLAS routines ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file define the lowering to BLAS operation of the SConv transform
// dialect extension.
//
//===----------------------------------------------------------------------===//
#include <cassert>

#include "SConv.h"
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::linalg;
using namespace mlir::LLVM;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::transform;

using LLVMCallOp = LLVM::CallOp;

#define DEBUG_TY "sconv-lowering"

#define DEBUG(expr)                                                            \
  LLVM_DEBUG(llvm::dbgs() << "[" << DEBUG_TY << "] " << expr << "\n")

//===----------------------------------------------------------------------===//
// Lowering to BLAS
//===----------------------------------------------------------------------===//

namespace {

/// Get an operation `T` from a module with the specified `name`.
template <typename T>
static std::optional<T> getOpByName(ModuleOp module, StringRef name) {
  for (auto func : module.getOps<T>()) {
    if (func.getName() == name)
      return func;
  }
  return {};
}

/// Create a LLVM function declaration for a BLAS micro-kernel. The function has
/// the following signature:
///
///   int(long m, long n, long k, float alpha, float *A, float *B, float *C,
///       long lda)
///
/// The name of the function comes from the `name` attribute in trasnform
/// operation.
static FailureOr<LLVMFuncOp> getOrCreateBlasFuncOp(TransformRewriter &rewriter,
                                                   LowerToBlasOp lowering,
                                                   ModuleOp module) {
  auto funcName = lowering.getName();

  // If the module already has a function with this name, use it.
  if (auto func = getOpByName<LLVMFuncOp>(module, funcName))
    return *func;

  // Add the new function to the end
  rewriter.setInsertionPointToEnd(module.getBody());

  auto ctx = module.getContext();
  auto loc = module.getLoc();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto f32Ty = Float32Type::get(ctx);
  auto ptrTy = LLVMPointerType::get(ctx);

  SmallVector<Type, 8> argumentTys = {
      /*m:*/ i64Ty, /*n:*/ i64Ty, /*k:*/ i64Ty, /*alpha:*/ f32Ty,
      /*A:*/ ptrTy, /*B:*/ ptrTy, /*C:*/ ptrTy, /*lda:*/ i64Ty,
  };

  auto returnTy = i32Ty;

  auto funcOp = rewriter.create<LLVMFuncOp>(
      loc, funcName, LLVMFunctionType::get(returnTy, argumentTys));

  return funcOp;
}

/// Given a value of type Memref, create a value of type LLVM pointer
static Value createPtrWithOffsetFromMemRef(OpBuilder builder, Value value,
                                           Twine &name) {
  assert(isa<MemRefType>(value.getType()) && "Expected memref value");

  auto ctx = value.getContext();
  auto loc = value.getLoc();
  Type i64Ty = builder.getI64Type();
  Type indTy = builder.getIndexType();
  Type ptrTy = LLVMPointerType::get(ctx);
  auto valTy = cast<MemRefType>(value.getType());

  value << name.concat("_tile");

  AffineExpr s_base, s_offset;
  bindSymbols(ctx, s_base, s_offset);

  // Mapping that computes the effective address of the tile. It is equal to the
  // base memref address plus the offset (in number of elements) times the
  // element size (in bytes).
  auto tileAddrMap = AffineMap::get(
      0, 2, s_base + s_offset * (valTy.getElementTypeBitWidth() / 8));

  // Create a `memref.extract_strided_metadata` to grab the offset
  Value offset =
      builder.create<ExtractStridedMetadataOp>(loc, value).getOffset();

  // Convert: memref<...> -> index
  Value base = builder.create<ExtractAlignedPointerAsIndexOp>(loc, value);
  base << name.concat("_base_ptr");

  // Compute effective address of the tile using the affine map
  Value index_ptr =
      builder.create<AffineApplyOp>(loc, tileAddrMap, ValueRange{base, offset});
  index_ptr << name.concat("_index_ptr");

  // Convert: index -> i64
  Value i64_ptr = builder.create<IndexCastOp>(loc, i64Ty, index_ptr);
  i64_ptr << name.concat("_i64_ptr");

  // Convert: i64 -> !llvm.ptr
  Value llvm_ptr = builder.create<IntToPtrOp>(loc, ptrTy, i64_ptr);
  llvm_ptr << name.concat("_llvm_ptr");

  return llvm_ptr;
}

/// Create a call to a BLAS microkernel (`callee`) that takes `memref` operands.
static FailureOr<LLVMCallOp> createBlasCallOp(TransformRewriter &rewriter,
                                              LLVMFuncOp callee,
                                              GenericOp ukernel) {
  auto ctx = ukernel.getContext();
  auto loc = ukernel.getLoc();

  // Get value and shape for the input operand
  Value inTile = ukernel->getOperand(0);
  auto inTileTy = dyn_cast<MemRefType>(inTile.getType());
  if (!inTileTy)
    inTile.getDefiningOp()->emitError()
        << "expected operand 0 to have type memref";
  ArrayRef<int64_t> inShape = inTileTy.getShape();

  // Get value and shape for the filter operand
  Value fsTile = ukernel->getOperand(1);
  auto fsTileTy = dyn_cast<MemRefType>(fsTile.getType());
  if (!fsTileTy)
    fsTile.getDefiningOp()->emitError()
        << "expected operand 1 to have type memref";
  ArrayRef<int64_t> fsShape = fsTileTy.getShape();

  // Get value and strides for the output operand
  Value outTile = ukernel->getOperand(2);
  auto outTileTy = dyn_cast<MemRefType>(outTile.getType());
  if (!outTileTy)
    outTile.getDefiningOp()->emitError()
        << "expected operand 2 to have type memref";
  auto [strides, _] = outTileTy.getStridesAndOffset();

  auto createI64 = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIntOp>(loc, value, 64);
  };

  auto createF32 = [&](double value) -> Value {
    return rewriter.create<arith::ConstantOp>(loc,
                                              rewriter.getF32FloatAttr(value));
  };

  auto createPtr = [&](Value source, Twine name) -> Value {
    return createPtrWithOffsetFromMemRef(rewriter, source, name);
  };

  rewriter.setInsertionPointAfter(ukernel);

  // Create arguments for the call
  Value m = createI64(inShape[2]) << "m";
  Value n = createI64(fsShape[1]) << "n";
  Value k = createI64(fsShape[0]) << "k";
  Value A = createPtr(inTile, "in");
  Value B = createPtr(fsTile, "fs");
  Value C = createPtr(outTile, "out");
  Value ldc = createI64(strides[1]) << "ldc";
  Value alpha = createF32(1.0) << "alpha";

  // Create call to BLAS micro-kernel
  auto call = rewriter.create<LLVMCallOp>(
      loc, callee, ValueRange{m, n, k, alpha, A, B, C, ldc});

  return call;
}

static FailureOr<Operation *> lowerKernelToBlas(TransformRewriter &rewriter,
                                                LowerToBlasOp lowering,
                                                GenericOp ukernel) {
  // Get function that contains the micro-kernel
  auto module = ukernel->getParentOfType<ModuleOp>();
  if (!module)
    return ukernel.emitError() << "payload must be inside a module";

  // Declare BLAS function after the current function
  auto callee = getOrCreateBlasFuncOp(rewriter, lowering, module);
  if (failed(callee))
    return ukernel.emitError()
           << "failed to create declaration to BLAS function";

  // Create a call to the BLAS function after the generic kernel
  auto call = createBlasCallOp(rewriter, *callee, ukernel);
  if (failed(call))
    return ukernel.emitError() << "failed to create call to BLAS function";

  // Transfer `schedule` attribute from the generic to the call
  if (auto schedule = ukernel->getAttr("schedule"))
    (*call)->setAttr("schedule", schedule);

  // Remove old kernel
  rewriter.eraseOp(ukernel);

  return call->getOperation();
}

} // namespace

DiagnosedSilenceableFailure LowerToBlasOp::apply(TransformRewriter &rewriter,
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

    auto result = lowerKernelToBlas(rewriter, *this, ukernel);
    if (failed(result))
      return DiagnosedSilenceableFailure::definiteFailure();

    resultOps.push_back(*result);
  }

  results.set((*this)->getOpResult(0), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void LowerToBlasOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult LowerToBlasOp::verify() {
  if (this->getName().size() == 0)
    return emitOpError("expected non-empty kernel name");
  return success();
}
