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
#include "xla/shape.h"

#include "shape.h"

extern "C" {
    void Shape_delete(Shape* s) {
        delete reinterpret_cast<xla::Shape*>(s);
    }

    int sizeof_Shape() {
        return sizeof(xla::Shape);
    }

    void set_array_Shape(Shape* arr, int idx, Shape* shape) {
        reinterpret_cast<xla::Shape*>(arr)[idx] = *reinterpret_cast<xla::Shape*>(shape);
    }
}
