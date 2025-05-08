/*
Copyright (C) 2025  Joel Berkeley

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
#include "xla/hlo/builder/lib/arithmetic.h"

#include "../xla_builder.h"

extern "C" {
    XlaOp* ArgMax(XlaOp& input, int output_type, int axis) {
        auto& input_ = reinterpret_cast<xla::XlaOp&>(input);
        xla::XlaOp res = xla::ArgMax(input_, (xla::PrimitiveType) output_type, axis);
        return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
    }
}
