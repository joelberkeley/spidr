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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
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

//class MemRefInsider
//    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};
//
//template <typename T>
//struct PtrElementModel
//    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
//          PtrElementModel<T>, T> {};

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
        printf("emitEnzymeADOp\n");
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

        ctx->loadDialect<mlir::enzyme::EnzymeDialect>();  // as suggested in MLIR tutorial
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

//        registry_.addExtension(+[](mlir::MLIRContext *ctx, mlir::LLVM::LLVMDialect *dialect) {
//            mlir::LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
//            mlir::LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
//            mlir::LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
//            mlir::LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
//            mlir::MemRefType::attachInterface<PtrElementModel<mlir::MemRefType>>(*ctx);
//            mlir::LLVM::LLVMStructType::attachInterface<
//                PtrElementModel<mlir::LLVM::LLVMStructType>>(*ctx);
//            mlir::LLVM::LLVMPointerType::attachInterface<
//                PtrElementModel<mlir::LLVM::LLVMPointerType>>(*ctx);
//            mlir::LLVM::LLVMArrayType::attachInterface<PtrElementModel<mlir::LLVM::LLVMArrayType>>(*ctx);
//        });

        mlir::transform::registerInterpreterPass();
        mlir::enzyme::registerGenerateApplyPatternsPass();
        mlir::enzyme::registerRemoveTransformPass();

        printf("module_op_.getOperation()\n");
        module_op_.getOperation()->dump();

        auto& region = module_op_.getOperation()->getRegion(0);
        auto& block = region.front();
        auto& operation = block.front();

        mlir::SymbolTable::setSymbolName(&operation, "tmp");
//        mlir::SymbolTable::setSymbolVisibility(&operation, mlir::SymbolTable::Visibility::Private);

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
        attrs.set("activity", activity);
        attrs.set("ret_activity", ret_activity);

        auto autodiff = mlir::Operation::create(
            mlir::UnknownLoc::get(ctx),
            mlir::OperationName("enzyme.autodiff", ctx),
            mlir::TypeRange({scalarf64}),
            mlir::ValueRange(entry_block->getArgument(0)),
            std::move(attrs),
//            mlir::NamedAttributeList({
//                mlir::NamedAttribute(mlir::StringAttr::get("fn", ctx), operation.getAttr("sym_name")),
//                mlir::NamedAttribute(mlir::StringAttr::get("activity", ctx), activity),
//                mlir::NamedAttribute(mlir::StringAttr::get("ret_activity", ctx), ret_activity),
//            }),
            mlir::OpaqueProperties(nullptr)
        );

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

        printf("module_op_.getOperation()\n");
        module_op_.getOperation()->dump();

        mlir::PassManager pm(ctx);
        printf("0\n");
        pm.addPass(mlir::enzyme::createDifferentiatePass());
        printf("1\n");
        pm.run(func_op);
        printf("2\n");

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(func_op));
    }
}
