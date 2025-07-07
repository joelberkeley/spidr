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
#include "mlir/IR/TypeRange.h"

#include "TypeRange.h"
#include "Types.h"

extern "C" {
    void TypeRange_delete(TypeRange* s) {
        delete reinterpret_cast<mlir::TypeRange*>(s);
    }

    TypeRange* TypeRange_new(Type* types, size_t types_len) {
        auto types_ = reinterpret_cast<mlir::Type*>(types);
        auto types_ar = llvm::ArrayRef(types_, types_len);
        return reinterpret_cast<TypeRange*>(new mlir::TypeRange(types_ar));
    }
}
