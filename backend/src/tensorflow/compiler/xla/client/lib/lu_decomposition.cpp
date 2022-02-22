/*
Copyright 2022 Joel Berkeley

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
#include "tensorflow/compiler/xla/client/lib/lu_decomposition.h"

#include "lu_decomposition.h"
#include "../xla_builder.h"

extern "C" {
    LuDecompositionResult* LuDecomposition(XlaOp& a) {
        auto& a_ = reinterpret_cast<xla::XlaOp&>(a);
        xla::LuDecompositionResult res = xla::LuDecomposition(a_);
        return (new LuDecompositionResult{
            reinterpret_cast<XlaOp*>(new xla::XlaOp(res.lu)),
            reinterpret_cast<XlaOp*>(new xla::XlaOp(res.pivots)),
            reinterpret_cast<XlaOp*>(new xla::XlaOp(res.permutation)),
        });
    }
}
