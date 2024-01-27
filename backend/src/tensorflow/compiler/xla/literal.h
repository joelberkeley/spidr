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
#include "shape.h"
#include "shape_util.h"

extern "C" {
    struct Literal;

    Literal* Literal_new(Shape& shape);

    void Literal_delete(Literal* lit);

    int Literal_Get_bool(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    );
    int Literal_Get_int(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    );
    double Literal_Get_double(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index
    );

    void Literal_Set_bool(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    );
    void Literal_Set_int(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, int value
    );
    void Literal_Set_double(
        Literal& lit, int* multi_index, int multi_index_len, ShapeIndex& shape_index, double value
    );
}
