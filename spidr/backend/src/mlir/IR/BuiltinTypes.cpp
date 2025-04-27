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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"

#include "BuiltinTypes.h"
#include "MLIRContext.h"
#include "TypeRange.h"
#include "Types.h"

extern "C" {
    void FloatType_delete(FloatType* s) {
        delete reinterpret_cast<mlir::FloatType*>(s);
    }

    void RankedTensorType_delete(RankedTensorType* s) {
        delete reinterpret_cast<mlir::RankedTensorType*>(s);
    }

    void set_array_RankedTensorType(Type* arr, size_t idx, RankedTensorType* value) {
        reinterpret_cast<mlir::Type*>(arr)[idx] = *reinterpret_cast<mlir::RankedTensorType*>(value);
    }

    void set_array_FloatType(Type* arr, size_t idx, FloatType* value) {
        reinterpret_cast<mlir::Type*>(arr)[idx] = *reinterpret_cast<mlir::FloatType*>(value);
    }

    struct FunctionType;

    void FunctionType_delete(FunctionType* s) {
        delete reinterpret_cast<mlir::FunctionType*>(s);
    }

    FunctionType* FunctionType_get(MLIRContext* ctx, TypeRange* inputs, TypeRange* results) {
        auto ctx_ = reinterpret_cast<mlir::MLIRContext*>(ctx);
        auto inputs_ = reinterpret_cast<mlir::TypeRange*>(inputs);
        auto results_ = reinterpret_cast<mlir::TypeRange*>(results);

        auto res = mlir::FunctionType::get(ctx_, *inputs_, *results_);
        return reinterpret_cast<FunctionType*>(new mlir::FunctionType(res));
    }

    RankedTensorType* RankedTensorType_get(int64_t* shape, size_t shape_len, Type& elementType) {
        auto elementType_ = reinterpret_cast<mlir::Type&>(elementType);
        llvm::ArrayRef<int64_t> shape_(shape, shape_len);
        auto res = mlir::RankedTensorType::get(shape_, elementType_);
        return reinterpret_cast<RankedTensorType*>(new mlir::RankedTensorType(res));
    }
}
