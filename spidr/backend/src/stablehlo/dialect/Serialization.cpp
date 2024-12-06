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
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/Version.h"

#include "../../mlir/IR/BuiltinOps.h"
#include "../../ffi.h"

extern "C" {
    bool serializePortableArtifact(ModuleOp& module, string& str) {
        auto& module_ = reinterpret_cast<mlir::ModuleOp&>(module);
        auto& str_ = reinterpret_cast<std::string&>(str);
        llvm::raw_string_ostream os(str_);
        mlir::BytecodeWriterConfig config;
        mlir::writeBytecodeToFile(module_, os, config);
        printf("%s\n", str_.c_str());
        auto version = mlir::vhlo::Version::getCurrentVersion().toString();
        return mlir::failed(mlir::stablehlo::serializePortableArtifact(module_, version, os));
    }
}
