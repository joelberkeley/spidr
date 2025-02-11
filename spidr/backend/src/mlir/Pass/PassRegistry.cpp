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
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include "PassManager.h"
#include "../../llvm/Support/raw_ostream.h"

extern "C" {
    int parsePassPipeline(char* pipeline, OpPassManager& pm, raw_ostream& errorStream) {
        auto& pm_ = reinterpret_cast<mlir::OpPassManager&>(pm);
        auto& errorStream_ = reinterpret_cast<llvm::raw_ostream&>(errorStream);
        return (int) mlir::parsePassPipeline(pipeline, pm_, errorStream_).succeeded();
    }
}
