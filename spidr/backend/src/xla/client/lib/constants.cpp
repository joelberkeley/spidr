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
#include "xla/client/lib/constants.h"

#include "../xla_builder.h"

XlaOp* constantOp(
    std::function<xla::XlaOp(xla::XlaBuilder*, xla::PrimitiveType)> op,
    XlaBuilder* builder,
    int type
) {
    auto builder_ = reinterpret_cast<xla::XlaBuilder*>(builder);
    xla::XlaOp res = op(builder_, (xla::PrimitiveType) type);
    return reinterpret_cast<XlaOp*>(new xla::XlaOp(res));
}

extern "C" {
    XlaOp* MinValue(XlaBuilder* builder, int type) {
        return constantOp(xla::MinValue, builder, type);
    }

    XlaOp* MinFiniteValue(XlaBuilder* builder, int type) {
        return constantOp(xla::MinFiniteValue, builder, type);
    }

    XlaOp* MaxValue(XlaBuilder* builder, int type) {
        return constantOp(xla::MaxValue, builder, type);
    }

    XlaOp* MaxFiniteValue(XlaBuilder* builder, int type) {
        return constantOp(xla::MaxFiniteValue, builder, type);
    }
}
