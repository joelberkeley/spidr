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

#include "BuiltinTypes.h"
#include "Types.h"

extern "C" {
    struct FloatType;

    void FloatType_delete(FloatType* s) {
        delete reinterpret_cast<mlir::FloatType*>(s);
    }

    FunctionType* FunctionType_get(
        MLIRContext* ctx, Type* inputs, size_t inputs_len, Type* results, size_t results_len
    ) {
        auto ctx_ = reinterpret_cast<mlir::MLIRContext*>(ctx);
        auto inputs_ = reinterpret_cast<mlir::Type*>(inputs);
        auto inputs_ar = llvm::ArrayRef(inputs_, inputs_len);
        auto results_ = reinterpret_cast<mlir::Type*>(results);
        auto results_ar = llvm::ArrayRef(results_, results_len);

        auto res = mlir::func::FunctionType::get(ctx_, inputs_ar, results_ar);
        return reinterpret_cast<FunctionType*>(new mlir::func::FunctionType(res));
    }

    RankedTensorType* RankedTensorType_get(int64_t* shape, size_t shape_len, Type& elementType) {
        auto elementType_ = reinterpret_cast<mlir::Type&>(elementType);
        llvm::ArrayRef<int64_t> shape_(shape, shape_len);
        auto res = mlir::RankedTensorType::get(shape_, elementType_);
        return reinterpret_cast<RankedTensorType*>(new mlir::RankedTensorType(res));
    }
}
