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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "BuiltinAttributes.h"
#include "BuiltinTypes.h"

extern "C" {
    void DenseElementsAttr_delete(DenseElementsAttr* s) {
        delete reinterpret_cast<mlir::DenseElementsAttr*>(s);
    }

    DenseElementsAttr* DenseElementsAttr_get(RankedTensorType& type, double value) {
        auto type_ = reinterpret_cast<mlir::RankedTensorType&>(type);
        auto res = mlir::DenseElementsAttr::get(type_, value);
        return reinterpret_cast<DenseElementsAttr*>(new mlir::DenseElementsAttr(res));
    }
}
