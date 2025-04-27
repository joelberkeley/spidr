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
#include "mlir/IR/Location.h"

#include "Location.h"
#include "MLIRContext.h"

extern "C" {
    Location* UnknownLoc_get(MLIRContext* context) {
        auto context_ = reinterpret_cast<mlir::MLIRContext*>(context);
        auto res = mlir::UnknownLoc::get(context_);
        return reinterpret_cast<Location*>(new mlir::Location(res));
    }
}
