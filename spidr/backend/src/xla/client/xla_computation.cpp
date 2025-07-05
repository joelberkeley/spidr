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
#include "xla/client/xla_computation.h"

#include "../../ffi.h"
#include "xla_computation.h"

extern "C" {
    void XlaComputation_delete(XlaComputation* s) {
        delete reinterpret_cast<xla::XlaComputation*>(s);
    }

    string* XlaComputation_SerializeAsString(XlaComputation* s) {
        auto s_ = reinterpret_cast<xla::XlaComputation*>(s);
        auto serialized = s_->proto().SerializeAsString();
        return reinterpret_cast<string*>(new std::string(serialized));
    }
}
