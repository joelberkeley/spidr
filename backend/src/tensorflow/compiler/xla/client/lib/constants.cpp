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
#include "tensorflow/compiler/xla/client/lib/constants.h"

#include "../xla_builder.h"

extern "C" {
    XlaOp* MinFiniteValue(XlaBuilder* builder, int type) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        xla::XlaOp res = xla::MinFiniteValue(builder_, (xla::PrimitiveType) type);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }

    XlaOp* MaxFiniteValue(XlaBuilder* builder, int type) {
        auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
        xla::XlaOp res = xla::MaxFiniteValue(builder_, (xla::PrimitiveType) type);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
