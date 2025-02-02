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

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
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



//#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
//#include "mlir/Dialect/Async/IR/Async.h"
//#include "mlir/Dialect/Complex/IR/Complex.h"
//#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
//#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
//#include "mlir/Dialect/GPU/IR/GPUDialect.h"
//#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
//#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
//#include "mlir/Dialect/Linalg/IR/Linalg.h"
//#include "mlir/Dialect/Math/IR/Math.h"
//#include "mlir/Dialect/MemRef/IR/MemRef.h"
//#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
//#include "mlir/Dialect/SCF/IR/SCF.h"
//#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

//#include "mlir/Dialect/Transform/IR/TransformDialect.h"

//class MemRefInsider
//    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};
//
//template <typename T>
//struct PtrElementModel
//    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<PtrElementModel<T>, T> {};

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
        mlir::DialectRegistry registry;

        ctx->loadDialect<mlir::arith::ArithDialect>();
        ctx->loadDialect<mlir::tensor::TensorDialect>();
        ctx->loadDialect<mlir::func::FuncDialect>();
        ctx->loadDialect<mlir::stablehlo::StablehloDialect>();
        ctx->loadDialect<mlir::chlo::ChloDialect>();
//        ctx->loadDialect<mlir::mhlo::MhloDialect>();

        registry.insert<mlir::stablehlo::StablehloDialect>();
        ctx->loadDialect<mlir::enzyme::EnzymeDialect>();
        registry.insert<mlir::enzyme::EnzymeDialect>();

        prepareRegistry(registry);

//  registry.insert<mlir::affine::AffineDialect>();
//  registry.insert<mlir::LLVM::LLVMDialect>();
//  registry.insert<mlir::memref::MemRefDialect>();
//  registry.insert<mlir::async::AsyncDialect>();
//  registry.insert<mlir::complex::ComplexDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::arith::ArithDialect>();
//  registry.insert<mlir::cf::ControlFlowDialect>();
//  registry.insert<mlir::scf::SCFDialect>();
//  registry.insert<mlir::gpu::GPUDialect>();
//  registry.insert<mlir::NVVM::NVVMDialect>();
//  registry.insert<mlir::omp::OpenMPDialect>();
//  registry.insert<mlir::math::MathDialect>();
//  registry.insert<mlir::linalg::LinalgDialect>();
//  registry.insert<mlir::DLTIDialect>();

        mlir::registerenzymePasses();
        mlir::registerCSEPass();
        mlir::registerCanonicalizerPass();
//  mlir::registerConvertAffineToStandardPass();
//  mlir::registerSCCPPass();
//  mlir::registerInlinerPass();
//  mlir::registerSymbolDCEPass();
//  mlir::registerLoopInvariantCodeMotionPass();
//  mlir::registerConvertSCFToOpenMPPass();
//  mlir::affine::registerAffinePasses();
//  mlir::registerReconcileUnrealizedCasts();

//  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::LLVM::LLVMDialect *dialect) {
//    mlir::LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
//    mlir::LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
//    mlir::LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
//    mlir::LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
//    mlir::MemRefType::attachInterface<PtrElementModel<mlir::MemRefType>>(*ctx);
//    mlir::LLVM::LLVMStructType::attachInterface<PtrElementModel<mlir::LLVM::LLVMStructType>>(*ctx);
//    mlir::LLVM::LLVMPointerType::attachInterface<PtrElementModel<mlir::LLVM::LLVMPointerType>>(*ctx);
//    mlir::LLVM::LLVMArrayType::attachInterface<PtrElementModel<mlir::LLVM::LLVMArrayType>>(*ctx);
//  });

        mlir::enzyme::registerCoreDialectAutodiffInterfaces(registry);
        ctx->appendDialectRegistry(registry);

        auto& region = module_op_.getOperation()->getRegion(0);
        auto& block = region.front();
        auto& operation = block.front();

        mlir::SymbolTable::setSymbolName(&operation, "tmp");

        auto scalarf64 = mlir::RankedTensorType::get({}, mlir::FloatType::getF64(ctx));
        auto func_type = mlir::FunctionType::get(ctx, {scalarf64}, {scalarf64});
        auto func_op = mlir::func::FuncOp::create(mlir::UnknownLoc::get(ctx), "main", func_type);

        block.push_back(func_op);

        auto entry_block = func_op.addEntryBlock();

        auto diff = mlir::OpBuilder(ctx).create<mlir::stablehlo::ConstantOp>(
            mlir::UnknownLoc::get(ctx), mlir::DenseElementsAttr::get(scalarf64, 1.0)
        );
        entry_block->push_back(diff);

        auto activity = mlir::enzyme::ActivityAttr::get(ctx, mlir::enzyme::Activity::enzyme_active);
        auto ret_activity = mlir::enzyme::ActivityAttr::get(
            ctx, mlir::enzyme::Activity::enzyme_activenoneed
        );

        // we can probably improve this by following the stablehlo example
        // https://github.com/openxla/stablehlo/blob/main/examples/c%2B%2B/ExampleAdd.cpp

        auto autodiff = mlir::OpBuilder(ctx).create<mlir::enzyme::AutoDiffOp>(
            mlir::UnknownLoc::get(ctx),
            mlir::TypeRange({scalarf64}),
            "tmp",
            mlir::ValueRange({entry_block->getArgument(0), diff->getOpResult(0)}),
            mlir::ArrayAttr::get(ctx, {activity}),
            mlir::ArrayAttr::get(ctx, {ret_activity})
        );

//        auto autodiff = mlir::Operation::create(
//            mlir::UnknownLoc::get(ctx),
//            mlir::OperationName("enzyme.autodiff", ctx),
//            mlir::TypeRange({scalarf64}),
//            mlir::ValueRange(entry_block->getArgument(0)),
//            {},
//            mlir::OpaqueProperties(nullptr)
//        );
//        autodiff->removeAttr("width");
//        autodiff->setDiscardableAttrs(attrs.getDictionary(ctx));

//        auto state = mlir::OperationState(mlir::UnknownLoc::get(ctx), "enzyme.autodiff");
//        state.addOperands(mlir::ValueRange(entry_block->getArgument(0)));
//        state.addTypes({scalarf64});
//        auto autodiff = mlir::Operation::create(state);
        entry_block->push_back(autodiff);

        auto return_op = mlir::OpBuilder(ctx).create<mlir::func::ReturnOp>(
            mlir::UnknownLoc::get(ctx),
            mlir::ValueRange(autodiff->getOpResult(0))
        );
        entry_block->push_back(return_op);

        mlir::PassManager pm(ctx);

        pm.addPass(mlir::enzyme::createDifferentiatePass());

//  auto pass = "enzyme-wrap{infn=main outfn= retTys=enzyme_activenoneed argTys=enzyme_active}, canonicalize, remove-unnecessary-enzyme-ops";
//  auto pass = "canonicalize, remove-unnecessary-enzyme-ops, arith-raise";

        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::enzyme::createRemoveUnusedEnzymeOpsPass());
        pm.addPass(mlir::enzyme::createArithRaisingPass());
        pm.run(module_op_);

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(module_op_));
    }
}
