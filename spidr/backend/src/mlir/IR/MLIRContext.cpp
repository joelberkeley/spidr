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
#include "mlir/IR/MLIRContext.h"

#include "DialectRegistry.h"
#include "MLIRContext.h"

extern "C" {
    MLIRContext* MLIRContext_new() {
        return reinterpret_cast<MLIRContext*>(new mlir::MLIRContext);
    }

    void MLIRContext_delete(MLIRContext* s) {
        delete reinterpret_cast<mlir::MLIRContext*>(s);
    }

//    DialectRegistry* MLIRContext_getDialectRegistry(MLIRContext& s) {
//        auto& s_ = reinterpret_cast<mlir::MLIRContext&>(s);
//        return reinterpret_cast<DialectRegistry*>(s_.getDialectRegistry());
//    }

    void MLIRContext_appendDialectRegistry(MLIRContext& s, DialectRegistry& registry) {
        auto& registry_ = reinterpret_cast<mlir::DialectRegistry&>(registry);
        reinterpret_cast<mlir::MLIRContext&>(s).appendDialectRegistry(registry_);
    }
}
