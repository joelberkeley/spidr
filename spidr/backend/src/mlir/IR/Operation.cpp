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
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // todo: extract to funcops

#include "Operation.h"
#include "Value.h"
#include "ValueRange.h"

extern "C" {
    void Operation_erase(Operation& s) {
        reinterpret_cast<mlir::Operation&>(s).erase();
    }

    struct CallOp;

    ResultRange* Operation_getOpResults(CallOp& s) {
        auto& s_ = reinterpret_cast<mlir::func::CallOp&>(s);
        auto res = s_.getOperation()->getOpResults();  // todo extract getOperation to funcops
        return reinterpret_cast<ResultRange*>(new mlir::ResultRange(res));
    }

    OpResult* Operation_getOpResult(CallOp& s, unsigned idx) {
        auto& s_ = reinterpret_cast<mlir::func::CallOp&>(s);
        auto res = s_.getOperation()->getOpResult(idx);  // todo extract getOperation to funcops
        return reinterpret_cast<OpResult*>(new mlir::OpResult(res));
    }
}
