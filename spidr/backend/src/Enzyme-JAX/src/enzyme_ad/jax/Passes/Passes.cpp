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
#include "../../../../../mlir/IR/BuiltinOps.h"
#include "../../../../../mlir/IR/DialectRegistry.h"
#include "../../../../../mlir/Pass/Pass.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

extern "C" {
    void regsiterenzymeXLAPasses_() {
        regsiterenzymeXLAPasses();
    }

    void registerenzymePasses() {
        mlir::registerenzymePasses();
    }

    Pass* createDifferentiatePass() {
        return reinterpret_cast<Pass*>(mlir::enzyme::createDifferentiatePass().release());
    }

     ModuleOp* emitEnzymeADOp(ModuleOp& module_op) {
        auto module_op_ = reinterpret_cast<mlir::ModuleOp&>(module_op);

        auto ctx = module_op_.getContext();
        mlir::DialectRegistry registry_;
        ctx->loadDialect<mlir::enzyme::EnzymeDialect>();
        registry_.insert<mlir::enzyme::EnzymeDialect>();
        ctx->appendDialectRegistry(registry_);

//        printf("module_op_.getOperation()\n");
        module_op_.getOperation()->dump();

        auto& region = module_op_.getOperation()->getRegion(0);
        auto& block = region.front();
        auto& operation = block.front();

        mlir::SymbolTable::setSymbolName(&operation, "tmp");

        auto scalarf64 = mlir::RankedTensorType::get({}, mlir::FloatType::getF64(ctx));
        auto func_type = mlir::FunctionType::get(ctx, {scalarf64}, {scalarf64});
        auto func_op = mlir::func::FuncOp::create(mlir::UnknownLoc::get(ctx), "main", func_type);

        block.push_back(func_op);

        auto entry_block = func_op.addEntryBlock();

        auto activity = mlir::enzyme::ActivityAttr::get(ctx, mlir::enzyme::Activity::enzyme_active);
        auto ret_activity = mlir::enzyme::ActivityAttr::get(
            ctx, mlir::enzyme::Activity::enzyme_activenoneed
        );

        mlir::NamedAttrList attrs;
        attrs.set("fn", operation.getAttr("sym_name"));
        attrs.set("activity", mlir::ArrayAttr::get(ctx, {activity}));
        attrs.set("ret_activity", mlir::ArrayAttr::get(ctx, {ret_activity}));

        auto autodiff = mlir::Operation::create(
            mlir::UnknownLoc::get(ctx),
            mlir::OperationName("enzyme.autodiff", ctx),
            mlir::TypeRange({scalarf64}),
            mlir::ValueRange(entry_block->getArgument(0)),
            {},
            mlir::OpaqueProperties(nullptr)
        );
        autodiff->removeAttr("width");
        autodiff->setDiscardableAttrs(attrs.getDictionary(ctx));

//        auto state = mlir::OperationState(mlir::UnknownLoc::get(ctx), "enzyme.autodiff");
//        state.addOperands(mlir::ValueRange(entry_block->getArgument(0)));
//        state.addTypes({scalarf64});
//        state.addAttribute("fn", operation.getAttr("sym_name"));
//        state.addAttribute("activity", {activity});
//        state.addAttribute("ret_activity", {ret_activity});
//        auto autodiff = mlir::Operation::create(state);
        entry_block->push_back(autodiff);

        auto return_op = mlir::OpBuilder(ctx).create<mlir::func::ReturnOp>(
            mlir::UnknownLoc::get(ctx),
            mlir::ValueRange(autodiff->getOpResult(0))
        );
        entry_block->push_back(return_op);

//        printf("module_op_.getOperation()\n");
        module_op_.getOperation()->dump();

        mlir::PassManager pm(ctx);
        printf("0\n");
        pm.addPass(mlir::enzyme::createDifferentiatePass());
        printf("1\n");
        pm.run(module_op_);
        printf("2\n");

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(module_op_));
    }
}
