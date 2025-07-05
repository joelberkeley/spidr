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
#include "mlir/Pass/PassManager.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "../../../../../mlir/Pass/PassManager.h"

extern "C" {
    void PassManager_addPass_ArithRaisingPass(PassManager& s) {
        auto& s_ = reinterpret_cast<mlir::PassManager&>(s);
        s_.addPass(mlir::enzyme::createArithRaisingPass());
    }
}
