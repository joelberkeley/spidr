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

extern "C" {
    struct Literal;

    Literal* Literal_new(Shape& shape);

    void Literal_delete(Literal* lit);

    int Literal_Get_bool(Literal& lit, int* indices);
    int Literal_Get_int(Literal& lit, int* indices);
    double Literal_Get_double(Literal& lit, int* indices);

    void Literal_Set_bool(Literal& lit, int* indices, int value);
    void Literal_Set_int(Literal& lit, int* indices, int value);
    void Literal_Set_double(Literal& lit, int* indices, double value);
}
