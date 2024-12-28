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
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OperationSupport.h"

#include "Attributes.h"
#include "Block.h"
#include "Location.h"
#include "ValueRange.h"

extern "C" {
    struct OperationState;

    OperationState* OperationState_new(Location& location, char* name) {
        auto& location_ = reinterpret_cast<mlir::Location&>(location);
        auto op_state = new mlir::OperationState(location_, name);
        return reinterpret_cast<OperationState*>(op_state);
    }

    void OperationState_delete(OperationState* s) {
        delete reinterpret_cast<mlir::OperationState*>(s);
    }

    void OperationState_addOperands(OperationState& s, ValueRange& newOperands) {
        auto& s_ = reinterpret_cast<mlir::OperationState&>(s);
        auto& newOperands_ = reinterpret_cast<mlir::ValueRange&>(newOperands);
        s_.addOperands(newOperands_);
    }

    void OperationState_addAttribute(OperationState& s, char* name, Attribute& attr) {
        auto& s_ = reinterpret_cast<mlir::OperationState&>(s);
        auto& attr_ = reinterpret_cast<mlir::Attribute&>(attr);
        s_.addAttribute(name, attr_);
    }

//    void OperationState_addSuccessors(OperationState& s, Block* successor) {
//        auto& s_ = reinterpret_cast<mlir::OperationState&>(s);
//        auto successor_ = reinterpret_cast<mlir::Block*>(successor);
//        s_.addSuccessors(successor_);
//    }
}
