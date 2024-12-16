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
#include "mlir/Pass/PassManager.h"

#include "../IR/BuiltinOps.h"
#include "../IR/MLIRContext.h"
#include "../IR/Operation.h"

extern "C" {
    struct PassManager;

    PassManager* PassManager_new(MLIRContext* ctx) {
        auto ctx_ = reinterpret_cast<mlir::MLIRContext*>(ctx);
        return reinterpret_cast<PassManager*>(new mlir::PassManager(ctx_));
    }

    void PassManager_delete(PassManager* s) {
        delete reinterpret_cast<mlir::PassManager*>(s);
    }

    int PassManager_run(PassManager& s, Operation* op) {
        auto& s_ = reinterpret_cast<mlir::PassManager&>(s);
        auto op_ = reinterpret_cast<mlir::Operation*>(op);
        return (int) s_.run(op_).succeeded();
    }
}
