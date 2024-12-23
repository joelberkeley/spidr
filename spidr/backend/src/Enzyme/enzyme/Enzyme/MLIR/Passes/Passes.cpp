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
#include "mlir/IR/Location.h"

#include "../../../../../mlir/Pass/Pass.h"

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
//#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
//#include "Enzyme/MLIR/PassDetails.h"

//#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "../../../../../mlir/IR/BuiltinOps.h"
#include "../../../../../mlir/IR/DialectRegistry.h"

//void registerStableHLODialectAutoDiffInterface(
//    DialectRegistry &registry) {
//  registry.addExtension(+[](MLIRContext *context,
//                            stablehlo::StablehloDialect *) {
//    registerInterfaces(context);
//
//    // SortOp::attachInterface<AutoDiffSort>(*context);
//
//    WhileOp::attachInterface<ADDataFlowWhileOp>(*context);
//    SortOp::attachInterface<ADDataFlowSortOp>(*context);
//    ScatterOp::attachInterface<ADDataFlowScatterOp>(*context);
//    ReduceOp::attachInterface<ADDataFlowReduceOp>(*context);
//
//    CaseOp::attachInterface<RegionBranchCaseOp>(*context);
//
//    ScatterOp::attachInterface<ScatterActivity>(*context);
//    ScatterOp::attachInterface<AutoDiffScatter>(*context);
//
//    ReturnOp::attachInterface<AutoDiffHLOReturn>(*context);
//
//    ReduceOp::attachInterface<AutoDiffReduceFwd<ReduceOp>>(*context);
//    IfOp::attachInterface<AutoDiffIfRev>(*context);
//    IfOp::attachInterface<AutoDiffIfFwd>(*context);
//    IfOp::attachInterface<AutoDiffIfCF>(*context);
//
//    WhileOp::attachInterface<AutoDiffWhileFwd>(*context);
//    WhileOp::attachInterface<AutoDiffWhileRev>(*context);
//    ReduceOp::attachInterface<AutoDiffReduceCF<ReduceOp>>(*context);
//    WhileOp::attachInterface<AutoDiffReduceCF<WhileOp>>(*context);
//    BroadcastInDimOp::attachInterface<AutoDiffBroadcastInDimRev>(*context);
//    SliceOp::attachInterface<AutoDiffSliceRev>(*context);
//    DynamicUpdateSliceOp::attachInterface<AutoDiffDynamicSliceUpdateRev>(
//        *context);
//    ReduceOp::attachInterface<AutoDiffReduceRev>(*context);
//    ConcatenateOp::attachInterface<AutoDiffConcatenateRev>(*context);
//
//    ConstantOp::attachInterface<SHLOConstantOpBatchInterface>(*context);
//    TransposeOp::attachInterface<SHLOTransposeOpBatchInterface>(*context);
//    IfOp::attachInterface<SHLOGenericBatchOpInterface<IfOp>>(*context);
//    WhileOp::attachInterface<SHLOGenericBatchOpInterface<WhileOp>>(*context);
//
//    ReverseOp::attachInterface<SHLOGenericBatchOpInterface<ReverseOp>>(
//        *context); // TODO: simpler version with newly named dims
//    ScatterOp::attachInterface<SHLOGenericBatchOpInterface<ScatterOp>>(
//        *context); // TODO: simpler version with newly named dims
//    ConvolutionOp::attachInterface<SHLOGenericBatchOpInterface<ConvolutionOp>>(
//        *context); // TODO: simpler version with newly named dims
//  });
//}

//void register_all(mlir::DialectRegistry& reg) {
//    registry.insert<mlir::affine::AffineDialect>();
//    registry.insert<mlir::LLVM::LLVMDialect>();
//    registry.insert<mlir::memref::MemRefDialect>();
//    registry.insert<mlir::async::AsyncDialect>();
//    registry.insert<mlir::tensor::TensorDialect>();
//    registry.insert<mlir::func::FuncDialect>();
//    registry.insert<mlir::arith::ArithDialect>();
//    registry.insert<mlir::cf::ControlFlowDialect>();
//    registry.insert<mlir::scf::SCFDialect>();
//    registry.insert<mlir::gpu::GPUDialect>();
//    registry.insert<mlir::NVVM::NVVMDialect>();
//    registry.insert<mlir::omp::OpenMPDialect>();
//    registry.insert<mlir::math::MathDialect>();
//    registry.insert<mlir::linalg::LinalgDialect>();
//    registry.insert<DLTIDialect>();
//    registry.insert<mlir::mhlo::MhloDialect>();
//    registry.insert<mlir::stablehlo::StablehloDialect>();
//    registry.insert<mlir::chlo::ChloDialect>();
//
//    registry.insert<mlir::enzyme::EnzymeDialect>();
//
//    mlir::registerenzymePasses();
//    regsiterenzymeXLAPasses();
//    mlir::enzyme::registerXLAAutoDiffInterfaces(registry);
//
//    mlir::func::registerInlinerExtension(registry);
//
//    // Register the standard passes we want.
//    mlir::registerCSEPass();
//    mlir::registerConvertAffineToStandardPass();
//    mlir::registerSCCPPass();
//    mlir::registerInlinerPass();
//    mlir::registerCanonicalizerPass();
//    mlir::registerSymbolDCEPass();
//    mlir::registerLoopInvariantCodeMotionPass();
//    mlir::registerConvertSCFToOpenMPPass();
//    mlir::affine::registerAffinePasses();
//    mlir::registerReconcileUnrealizedCasts();
//
//    mlir::registerLLVMDialectImport(registry);
//    mlir::registerNVVMDialectImport(registry);
//
//    mlir::LLVM::registerInlinerInterface(registry);
//
//    /*
//    registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
//    LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
//    LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
//    LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
//    LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
//    MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
//    LLVM::LLVMStructType::attachInterface<
//        PtrElementModel<LLVM::LLVMStructType>>(*ctx);
//    LLVM::LLVMPointerType::attachInterface<
//        PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
//    LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
//        *ctx);
//    });
//    */
//
//    // Register the autodiff interface implementations for upstream dialects.
//    enzyme::registerCoreDialectAutodiffInterfaces(registry);
//
//    // Transform dialect and extensions.
//    mlir::transform::registerInterpreterPass();
//    mlir::linalg::registerTransformDialectExtension(registry);
//    mlir::enzyme::registerGenerateApplyPatternsPass();
//    mlir::enzyme::registerRemoveTransformPass();
//    mlir::enzyme::registerEnzymeJaxTransformExtension(registry);
//}

extern "C" {
    Pass* createDifferentiatePass() {
        return reinterpret_cast<Pass*>(mlir::enzyme::createDifferentiatePass().release());
    }

    // doesn't belong here
    ModuleOp* emitEnzymeADOp(ModuleOp& module_op, DialectRegistry& registry) {
        auto& registry_ = reinterpret_cast<mlir::DialectRegistry&>(registry);

        mlir::enzyme::registerCoreDialectAutodiffInterfaces(registry_);
        // registerXLAAutoDiffInterfaces(registry_);
        // mlir::linalg::registerTransformDialectExtension(registry_);  // how to import?
        // mlir::enzyme::registerEnzymeJaxTransformExtension(registry_); // not public
        // mlir::func::registerInlinerExtension(registry_);  // not tried

        auto module_op_ = reinterpret_cast<mlir::ModuleOp&>(module_op);

        auto ctx = module_op_.getContext();
        auto state = mlir::OperationState(mlir::UnknownLoc::get(ctx), "enzyme.autodiff");

        auto scalarf64 = mlir::RankedTensorType::get({}, mlir::FloatType::getF64(ctx));
        state.addTypes({scalarf64});

        auto operands = module_op_.getOperation()->getOperands();  // complete guess
        state.addOperands(mlir::ValueRange(operands));

        auto operation = module_op_.getOperation();  // complete guess
        state.addAttribute("fn", operation->getAttr("sym_name"));
        auto activity = mlir::enzyme::ActivityAttr::get(ctx, mlir::enzyme::Activity::enzyme_active);
        state.addAttribute("activity", {activity}); // mlir::enzyme::Activity::enzyme_active
        auto ret_activity = mlir::enzyme::ActivityAttr::get(
            ctx, mlir::enzyme::Activity::enzyme_active
        );
        state.addAttribute("ret_activity", {ret_activity}); // mlir::enzyme::Activity::enzyme_activenoneed

        auto res = mlir::Operation::create(state);

        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(res));
    }
}
