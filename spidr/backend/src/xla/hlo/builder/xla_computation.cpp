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
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"

#include "../../../ffi.h"
#include "../../service/hlo.proto.h"
#include "../../shape.h"
#include "xla_computation.h"

extern "C" {
    XlaComputation* XlaComputation_new(HloModuleProto& proto) {
        auto& proto_ = reinterpret_cast<xla::HloModuleProto&>(proto);
        // this moves the proto? should we then not GC it?
        return reinterpret_cast<XlaComputation*>(new xla::XlaComputation(proto_));
    }

    void XlaComputation_delete(XlaComputation* s) {
        delete reinterpret_cast<xla::XlaComputation*>(s);
    }

    HloModuleProto* XlaComputation_proto(XlaComputation* s) {
        auto s_ = reinterpret_cast<xla::XlaComputation*>(s);
        return reinterpret_cast<HloModuleProto*>(new xla::HloModuleProto(s_->proto()));
    }
}
