//===-- SConv.h - Transform dialect Extension SConv -------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// @brief MLIR Transform dialect extension for the Sliced Convolution (SConv).
//
// WHAT
//   This header declares the SConv Transform extension and its custom ops
//   (generated via ODS). SConv takes linalg 2D convolutions (NCHW/FCHW) and
//   lowers them to tiled/packed linalg.generic micro-kernels guided by a
//   Convolution Slicing Analysis (CSA).
//
// WHY
//   The goal is to expose a reusable, analyzable transformation pipeline that
//   captures tiling, edge-case splitting, packing and (optionally) multipacking
//   in a declarative way (Transform dialect).
//
// HOW (high-level)
//   - Named convs → normalized linalg.generic with collapsed H×W.
//   - CSA decides schedule (IS/WS) and tile sizes (Nc, K2, K3).
//   - Edge-case splits (input/filter domains and CSA remainders).
//   - Two-level tiling and packing (plus multipacking when applicable).
//   - Affine maps implement the packing equations in the paper (Section 4).
//
// Usage (Transform interpreter)
//   registerSConv(registry);    // during tool initialization
//   // In a .mlir transform file: transform.structured.sconv ...
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef SCONV_H
#define SCONV_H

#include "mlir/Support/LLVM.h"
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Transform/IR/TransformAttrs.h>
#include <mlir/Dialect/Transform/IR/TransformDialect.h>
#include <mlir/Dialect/Transform/Interfaces/TransformInterfaces.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>

namespace mlir {
class CallOpInterface;
class RewriterBase;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace transform {
class AnyOpType;
class AnyValueType;
class OperationType;
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "SConv.h.inc"

// Register SConv Transform dialect extension.
void registerSConv(::mlir::DialectRegistry &registry);

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

using namespace mlir;

/// Create a named location from a name and another location.
static inline Location CreateNameLoc(const Twine &name, Location loc) {
#ifndef NDEBUG
  return NameLoc::get(StringAttr::get(loc.getContext(), name), loc);
#else
  return loc;
#endif
}

/// Set named location to an existing value
static inline Value SetNameLoc(Value value, const Twine &name) {
#ifndef NDEBUG
  value.setLoc(CreateNameLoc(name, value.getLoc()));
#endif
  return value;
}

/// Set named location to an existing op with a single result
static inline Operation *SetNameLoc(Operation *op, const Twine &name) {
#ifndef NDEBUG
  assert(op->getNumResults() == 1 && "Operation must have a single result");
  SetNameLoc(op->getResult(0), name);
#endif
  return op;
}

} // namespace

static inline mlir::Location operator<<(mlir::Location loc,
                                        const mlir::Twine &name) {
  return CreateNameLoc(name, loc);
}

static inline mlir::Value operator<<(mlir::Value value,
                                     const mlir::Twine &name) {
  return SetNameLoc(value, name);
}

static inline mlir::Operation *operator<<(mlir::Operation *op,
                                          const mlir::Twine &name) {
  return SetNameLoc(op, name);
}

#endif // SCONV_H
