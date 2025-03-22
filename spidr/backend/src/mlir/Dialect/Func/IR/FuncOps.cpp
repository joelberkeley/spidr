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
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "../../../IR/Block.h"
#include "../../../IR/BuiltinDialectBytecode.h"
#include "../../../IR/Location.h"

extern "C" {
    struct FuncOp;

    FuncOp* FuncOp_create(Location& location, char* name, FunctionType& type) {
        auto location_ = reinterpret_cast<mlir::Location&>(location);
        auto type_ = reinterpret_cast<mlir::FunctionType&>(type);
        auto res = mlir::func::FuncOp::create(location_, name, type_);
        return reinterpret_cast<FuncOp*>(new mlir::func::FuncOp(res));
    }

    Block* FuncOp_addEntryBlock(FuncOp& s) {
        auto s_ = reinterpret_cast<mlir::func::FuncOp&>(s);
        return reinterpret_cast<Block*>(s_.addEntryBlock());
    }
}
