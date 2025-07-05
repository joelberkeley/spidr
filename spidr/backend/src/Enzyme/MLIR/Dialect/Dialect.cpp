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
#include "mlir/IR/DialectRegistry.h"
#include "Enzyme/MLIR/Dialect/Dialect.h"

#include "../../../mlir/IR/DialectRegistry.h"
#include "../../../mlir/IR/MLIRContext.h"

extern "C" {
    void DialectRegistry_insert_EnzymeDialect(DialectRegistry& s) {
        reinterpret_cast<mlir::DialectRegistry&>(s).insert<mlir::enzyme::EnzymeDialect>();
    }

    void MLIRContext_loadDialect_EnzymeDialect(MLIRContext& s) {
        reinterpret_cast<mlir::MLIRContext&>(s).loadDialect<mlir::enzyme::EnzymeDialect>();
    }
}
