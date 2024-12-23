/*
Copyright 2024 Joel Berkeley

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
#include "Enzyme/MLIR/Passes/Passes.h"
#include "mlir/IR/BuiltinOps.h"

#include "../../../../../mlir/Pass/Pass.h"

// for AD function
#include "mlir/IR/BuiltinTypes.h"

#include "../../../../../mlir/IR/BuiltinOps.h"

extern "C" {
    Pass* createDifferentiatePass() {
        return reinterpret_cast<Pass*>(mlir::enzyme::createDifferentiatePass().release());
    }

    ModuleOp* emitEnzymeADOp(ModuleOp& module_op) {
        auto module_op_ = reinterpret_cast<mlir::ModuleOp&>(module_op);
        auto state = mlir::OperationState("enzyme.autodiff", mlir::Location());
        auto ctx = module_op_.getContext();

        auto scalarf64 = mlir::RankedTensorType::Builder()
            .setShape({})
            .setElementType(mlir::FloatType::getF64(ctx));
        state.addTypes({scalarf64});

        auto operand = module_op_.getOperand({0})  // complete guess
        state.addOperands(ValueRange({operand}));

        auto operation = module_op_.front();  // complete guess
        state.addAttribute("fn", operation.getAttr("sym_name"));
        auto activity = mlir::enzyme::ActivityAttr::get(ctx, mlir::enzyme::Activity::enzyme_active);
        state.addAttribute("activity", {activity}); // mlir::enzyme::Activity::enzyme_active
        auto ret_activity = mlir::enzyme::ActivityAttr::get(
            ctx, mlir::enzyme::Activity::enzyme_active
        );
        state.addAttribute("ret_activity", {ret_activity}); // mlir::enzyme::Activity::enzyme_activenoneed

        auto res = mlir::ModuleOp(&mlir::Builder(module_op_), state);  // complete guess

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(res));
    }
}
