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

#include "Value.h"

extern "C" {
    struct Value;

    size_t sizeof_Value() {
        return sizeof(mlir::Value);
    }

    void set_array_BlockArgument(Value* arr, size_t idx, BlockArgument* value) {
        reinterpret_cast<mlir::Value*>(arr)[idx] = *reinterpret_cast<mlir::BlockArgument*>(value);
    }

    void set_array_OpResult(Value* arr, size_t idx, OpResult* value) {
        reinterpret_cast<mlir::Value*>(arr)[idx] = *reinterpret_cast<mlir::OpResult*>(value);
    }

    void BlockArgument_delete(BlockArgument* s) {
        delete reinterpret_cast<mlir::BlockArgument*>(s);
    }

    void OpResult_delete(OpResult* s) {
        delete reinterpret_cast<mlir::OpResult*>(s);
    }
}
