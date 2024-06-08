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
#include "xla/client/lib/arithmetic.h"

#include "../xla_builder.h"

extern "C" {
    XlaOp* ArgMax(XlaOp& input, int output_type, int axis) {
        auto& input_ = reinterpret_cast<xla::XlaOp&>(input);
        xla::XlaOp res = xla::ArgMax(input_, (xla::PrimitiveType) output_type, axis);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* ArgMin(XlaOp& input, int output_type, int axis) {
        auto& input_ = reinterpret_cast<xla::XlaOp&>(input);
        xla::XlaOp res = xla::ArgMin(input_, (xla::PrimitiveType) output_type, axis);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
