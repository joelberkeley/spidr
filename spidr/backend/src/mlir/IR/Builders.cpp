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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Block.h"
#include "Builders.h"
#include "BuiltinTypes.h"
#include "MLIRContext.h"

extern "C" {
    struct Builder;

    FloatType* Builder_getF64Type(Builder& s) {
        auto s_ = reinterpret_cast<mlir::OpBuilder&>(s);
        auto res = s_.getF64Type();
        return reinterpret_cast<FloatType*>(new mlir::FloatType(res));
    }

    struct OpBuilder;

    OpBuilder* OpBuilder_new(MLIRContext* ctx) {
        auto ctx_ = reinterpret_cast<mlir::MLIRContext*>(ctx);
        return reinterpret_cast<OpBuilder*>(new mlir::OpBuilder(ctx_));
    }

    void OpBuilder_delete(OpBuilder* s) {
        delete reinterpret_cast<mlir::OpBuilder*>(s);
    }

    OpBuilder* OpBuilder_atBlockEnd(Block* block) {
        auto block_ = reinterpret_cast<mlir::Block*>(block);
        auto builder = mlir::OpBuilder::atBlockEnd(block_);
        return reinterpret_cast<OpBuilder*>(new mlir::OpBuilder(builder));
    }
}
