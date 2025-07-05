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
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "../../../IR/Block.h"
#include "../../../IR/Builders.h"
#include "../../../IR/BuiltinDialectBytecode.h"
#include "../../../IR/Location.h"
#include "../../../IR/Operation.h"
#include "../../../IR/TypeRange.h"
#include "../../../IR/ValueRange.h"

extern "C" {
    struct CallOp;

    void CallOp_delete(CallOp* s) {
        delete reinterpret_cast<mlir::func::CallOp*>(s);
    }

    CallOp* OpBuilder_create_CallOp(
        OpBuilder& s, Location& location, char* name, TypeRange& types, ValueRange& operands
    ) {
        auto s_ = reinterpret_cast<mlir::OpBuilder&>(s);
        auto location_ = reinterpret_cast<mlir::Location&>(location);
        auto types_ = reinterpret_cast<mlir::TypeRange&>(types);
        auto operands_ = reinterpret_cast<mlir::ValueRange&>(operands);
        auto res = s_.create<mlir::func::CallOp>(location_, name, types_, operands_);
        return reinterpret_cast<CallOp*>(new mlir::func::CallOp(res));
    }

    Operation* CallOp_getOperation(CallOp& s) {
        auto& s_ = reinterpret_cast<mlir::func::CallOp&>(s);
        auto res = s_.getOperation();
        return reinterpret_cast<Operation*>(res);
    }

    struct FuncOp;

    void FuncOp_delete(FuncOp* s) {
        delete reinterpret_cast<mlir::func::FuncOp*>(s);
    }

    Operation* FuncOp_getOperation(FuncOp& s) {
        auto& s_ = reinterpret_cast<mlir::func::FuncOp&>(s);
        auto res = s_.getOperation();
        return reinterpret_cast<Operation*>(res);
    }

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

    struct ReturnOp;

    void ReturnOp_delete(ReturnOp* s) {
        delete reinterpret_cast<mlir::func::ReturnOp*>(s);
    }

    ReturnOp* OpBuilder_create_ReturnOp(OpBuilder& s, Location& location, ResultRange& results) {
        auto s_ = reinterpret_cast<mlir::OpBuilder&>(s);
        auto location_ = reinterpret_cast<mlir::Location&>(location);
        auto results_ = reinterpret_cast<mlir::ResultRange&>(results);
        auto res = s_.create<mlir::func::ReturnOp>(location_, results_);
        return reinterpret_cast<ReturnOp*>(new mlir::func::ReturnOp(res));
    }
}
