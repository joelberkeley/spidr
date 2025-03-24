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
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"

#include "Operation.h"
#include "Value.h"
#include "ValueRange.h"

extern "C" {
    void Operation_erase(Operation& s) {
        reinterpret_cast<mlir::Operation&>(s).erase();
    }

    ResultRange* Operation_getOpResults(Operation& s) {
        auto& s_ = reinterpret_cast<mlir::Operation&>(s);
        auto res = s_.getOpResults();
        return reinterpret_cast<ResultRange*>(new mlir::ResultRange(res));
    }

    OpResult* Operation_getOpResult(Operation& s, unsigned idx) {
        auto& s_ = reinterpret_cast<mlir::Operation&>(s);
        auto res = s_.getOpResult(idx);
        return reinterpret_cast<OpResult*>(new mlir::OpResult(res));
    }
}
