/*
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

    string* XlaComputation_SerializeAsString(XlaComputation* s) {
        auto s_ = reinterpret_cast<xla::XlaComputation*>(s);
        auto serialized = s_->proto().SerializeAsString();
        return reinterpret_cast<string*>(new std::string(serialized));
    }
}
