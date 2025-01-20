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
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/tests/CheckOps.h"

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

extern "C" {
//    void regsiterenzymeXLAPasses_() {
//        regsiterenzymeXLAPasses();
//    }
//
//    void registerenzymePasses() {
//        mlir::registerenzymePasses();
//    }
//
////    Pass* createDifferentiatePass() {
////        return reinterpret_cast<Pass*>(mlir::enzyme::createDifferentiatePass().release());
////    }

     ModuleOp* emitEnzymeADOp(ModuleOp& module_op) {
//    xla::XlaBuilder builder("root");
//    auto xlaScalarf64 = xla::ShapeUtil::MakeScalarShape((xla::PrimitiveType) 12);
//    auto arg = xla::Parameter(&builder, 0, xlaScalarf64, "arg");
//    auto proto = builder.Build(xla::Square(arg))->proto();
//
//    mlir::MLIRContext ctx;
//    mlir::DialectRegistry registry_;
//    ctx.appendDialectRegistry(registry_);
//    mlir::mhlo::registerAllMhloDialects(registry_);
//    mlir::stablehlo::registerAllDialects(registry_);
//
//        auto module_op_ = xla::ConvertHloToStablehlo(ctx, &proto).value().release();
        auto module_op_ = reinterpret_cast<mlir::ModuleOp&>(module_op);
        auto ctx = module_op_.getContext();
        mlir::DialectRegistry registry_;

        ctx.loadDialect<mlir::enzyme::EnzymeDialect>();  // as suggested in MLIR tutorial
        registry_.insert<mlir::enzyme::EnzymeDialect>();
        registry_.insert<mlir::stablehlo::check::CheckDialect>();
        prepareRegistry(registry_);

        ctx->appendDialectRegistry(registry_);

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        mlir::registerenzymePasses();
        regsiterenzymeXLAPasses();

        mlir::registerCSEPass();
        mlir::registerConvertAffineToStandardPass();
        mlir::registerSCCPPass();
        mlir::registerInlinerPass();
        mlir::registerCanonicalizerPass();
        mlir::registerSymbolDCEPass();
        mlir::registerLoopInvariantCodeMotionPass();
        mlir::registerConvertSCFToOpenMPPass();
        mlir::affine::registerAffinePasses();
        mlir::registerReconcileUnrealizedCasts();

        registry_.addExtension(+[](mlir::MLIRContext *ctx, mlir::LLVM::LLVMDialect *dialect) {
            mlir::LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
            mlir::LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
            mlir::LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
            mlir::LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
            mlir::MemRefType::attachInterface<PtrElementModel<mlir::MemRefType>>(*ctx);
            mlir::LLVM::LLVMStructType::attachInterface<
                PtrElementModel<mlir::LLVM::LLVMStructType>>(*ctx);
            mlir::LLVM::LLVMPointerType::attachInterface<
                PtrElementModel<mlir::LLVM::LLVMPointerType>>(*ctx);
            mlir::LLVM::LLVMArrayType::attachInterface<PtrElementModel<mlir::LLVM::LLVMArrayType>>(*ctx);
        });

        mlir::transform::registerInterpreterPass();
        mlir::enzyme::registerGenerateApplyPatternsPass();
        mlir::enzyme::registerRemoveTransformPass();

        auto state = mlir::OperationState(mlir::UnknownLoc::get(ctx), "enzyme.autodiff");

        auto scalarf64 = mlir::RankedTensorType::get({}, mlir::FloatType::getF64(ctx));
        state.addTypes({scalarf64});

        auto operands = module_op_.getOperation()->getOperands();  // complete guess
        state.addOperands(mlir::ValueRange(operands));

        auto operation = module_op_.getOperation();  // complete guess
        state.addAttribute("fn", operation->getAttr("sym_name"));
        auto activity = mlir::enzyme::ActivityAttr::get(ctx, mlir::enzyme::Activity::enzyme_active);
        state.addAttribute("activity", {activity});
        auto ret_activity = mlir::enzyme::ActivityAttr::get(
            ctx, mlir::enzyme::Activity::enzyme_activenoneed
        );
        state.addAttribute("ret_activity", {ret_activity});

        auto res = mlir::Operation::create(state);

//    return 0;

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(res));
    }
}
