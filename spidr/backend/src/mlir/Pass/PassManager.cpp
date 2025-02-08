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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "Pass.h"
#include "PassManager.h"
#include "../IR/BuiltinOps.h"
#include "../IR/MLIRContext.h"
#include "../IR/Operation.h"

extern "C" {
    PassManager* PassManager_new(MLIRContext* ctx) {
        auto ctx_ = reinterpret_cast<mlir::MLIRContext*>(ctx);
        return reinterpret_cast<PassManager*>(new mlir::PassManager(ctx_));
    }

    void PassManager_delete(PassManager* s) {
        delete reinterpret_cast<mlir::PassManager*>(s);
    }

    void PassManager_addPass(PassManager& s, Pass* pass) {
        auto& s_ = reinterpret_cast<mlir::PassManager&>(s);
        auto pass_ = reinterpret_cast<mlir::Pass*>(pass);
        s_.addPass(std::make_unique<mlir::Pass>(pass_));  // will this work with concrete passes?
    }

    int PassManager_run(PassManager& s, Operation* op) {
        auto& s_ = reinterpret_cast<mlir::PassManager&>(s);
        auto op_ = reinterpret_cast<mlir::Operation*>(op);
        return (int) s_.run(op_).succeeded();
    }
}
