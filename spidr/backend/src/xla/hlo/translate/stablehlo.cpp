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
#include "mlir/IR/BuiltinOps.h"
#include "xla/service/hlo.pb.h"
#include "xla/hlo/translate/stablehlo.h"

#include "../../service/hlo.proto.h"
#include "../../../mlir/IR/BuiltinOps.h"
#include "../../../mlir/IR/MLIRContext.h"

extern "C" {
    ModuleOp* ConvertHloToStablehlo(MLIRContext& ctx, HloModuleProto* hlo_module) {
        auto& ctx_ = reinterpret_cast<mlir::MLIRContext&>(ctx);
        auto hlo_module_ = reinterpret_cast<xla::HloModuleProto*>(hlo_module);
        auto module_op = xla::ConvertHloToStablehlo(ctx_, hlo_module_);
        module_op.value()->dump();
        return reinterpret_cast<ModuleOp*>(new mlir::ModuleOp(module_op.value().release()));
    }

    HloModuleProto* ConvertStablehloToHlo(ModuleOp& module) {
        auto& module_ = reinterpret_cast<mlir::ModuleOp&>(module);
        auto hlo = xla::ConvertStablehloToHlo(module_).value().release();
        // mode ToProto to separate function?
        auto res = hlo->ToProto();
        return reinterpret_cast<HloModuleProto*>(new xla::HloModuleProto(res));
    }
}
