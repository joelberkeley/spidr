/*
Copyright 2025 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "stablehlo/dialect/StablehloOps.h"

#include "../../mlir/IR/Builders.h"
#include "../../mlir/IR/BuiltinAttributes.h"
#include "../../mlir/IR/Location.h"
#include "../../mlir/IR/Operation.h"

extern "C" {
    struct ConstantOp;

    void ConstantOp_delete(ConstantOp* s) {
        delete reinterpret_cast<mlir::stablehlo::ConstantOp*>(s);
    }

    ConstantOp* OpBuilder_create_ConstantOp(
        OpBuilder& s, Location& location, DenseElementsAttr& attr
    ) {
        auto s_ = reinterpret_cast<mlir::OpBuilder&>(s);
        auto location_ = reinterpret_cast<mlir::Location&>(location);
        auto attr_ = reinterpret_cast<mlir::DenseElementsAttr&>(attr);

        auto res = s_.create<mlir::stablehlo::ConstantOp>(location_, attr_);
        return reinterpret_cast<ConstantOp*>(new mlir::stablehlo::ConstantOp(res));
    }

    Operation* ConstantOp_getOperation(ConstantOp& s) {
        auto& s_ = reinterpret_cast<mlir::stablehlo::ConstantOp&>(s);
        auto res = s_.getOperation();
        return reinterpret_cast<Operation*>(res);
    }
}
