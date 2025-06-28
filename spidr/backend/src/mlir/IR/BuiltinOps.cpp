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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // todo

#include "BuiltinOps.h"
#include "MLIRContext.h"
#include "Operation.h"

extern "C" {
    void ModuleOp_delete(ModuleOp* s) {
        delete reinterpret_cast<mlir::ModuleOp*>(s);
    }

    void ModuleOp_dump(ModuleOp& s) {
        reinterpret_cast<mlir::ModuleOp&>(s).dump();
    }

    Operation* ModuleOp_getOperation(ModuleOp& s) {
        auto s_ = reinterpret_cast<mlir::ModuleOp&>(s);
        return reinterpret_cast<Operation*>(s_.getOperation());
    }

    void ModuleOp_push_back(ModuleOp& s, Operation* op) {
        auto s_ = reinterpret_cast<mlir::ModuleOp&>(s);
        auto op_ = reinterpret_cast<mlir::Operation*>(op);
        s_.push_back(op_);
    }
}
