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
#include "absl/status/status.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"

#include "../../service/hlo.proto.h"
#include "../../status.h"

extern "C" {
    struct ModuleOp;

    // i've skipped so much in this function it's almost unrecognisable
    ModuleOp* ConvertHloToStableHLO(
        // ModuleOp* module,  // there's no point in doing this the long way round when this is a workaround
        HloModuleProto const* hlo_module,
        // bool import_all_computations,
        // bool flatten_computation_args_result
    ) {
        // mostly copied from PyXlaComputationToMlirModule
        mlir::MLIRContext context;
        auto module = llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context));
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::mhlo::MhloDialect>();
        mlir::DialectRegistry registry;
        mlir::func::registerAllExtensions(registry);
        context.appendDialectRegistry(registry);

        auto res = new absl::Status;
        res = xla::ConvertHloToMlirHlo(
            module,
            reinterpret_cast<xla::HloModuleProto const*>(hlo_module),
            import_all_computations,
            flatten_computation_args_result,
        )

        mlir::PassManager pm(&context);
        pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
        return reinterpret_cast<ModuleOp*>(module);
    }

absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertMlirHloToHloModule(
    mlir::ModuleOp module, MlirToHloConversionOptions options = {});

    HloModule* ConvertMlirHloToHloModule (ModuleOp* module) {
        return mlir::
    }
}
