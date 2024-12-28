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
#include "stablehlo/dialect/Serialization.h"

#include "../../mlir/IR/BuiltinOps.h"
#include "../../llvm/Support/raw_ostream.h"
#include "../../ffi.h"

extern "C" {
    int serializePortableArtifact(ModuleOp& module, string& version, raw_ostream& os) {
        auto& module_ = reinterpret_cast<mlir::ModuleOp&>(module);
        auto& version_ = reinterpret_cast<std::string&>(version);
        auto& os_ = reinterpret_cast<llvm::raw_ostream&>(os);
        auto result = mlir::stablehlo::serializePortableArtifact(module_, version_, os_);
        return (int) result.succeeded();
    }
}
