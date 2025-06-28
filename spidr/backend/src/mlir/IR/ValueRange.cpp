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
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "Value.h"
#include "ValueRange.h"

extern "C" {
    void ValueRange_delete(ValueRange* s) {
        delete reinterpret_cast<mlir::ValueRange*>(s);
    }

    ValueRange* ValueRange_new(Value* values, size_t values_len) {
        auto values_ = reinterpret_cast<mlir::Value*>(values);
        auto values_ar = llvm::ArrayRef(values_, values_len);
        return reinterpret_cast<ValueRange*>(new mlir::ValueRange(values_ar));
    }

    void ResultRange_delete(ResultRange* s) {
        delete reinterpret_cast<mlir::ResultRange*>(s);
    }
}
