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
#include <algorithm>
#include <cstring>
#include <string>

#include "tensorflow/compiler/xla/shape.h"

#include "shape.h"

extern "C" {
    const char* c_string_copy(std::string str) {
        char *res = NULL;
        auto len = str.length();
        res = (char *) malloc(len + 1);
        strncpy(res, str.c_str(), len);
        res[len] = '\0';
        return res;
    }

    void Shape_delete(Shape* s) {
        delete reinterpret_cast<xla::Shape*>(s);
    }

    const char* Shape_DebugString(Shape& s) {
        auto& s_ = reinterpret_cast<xla::Shape&>(s);
        return c_string_copy(s_.DebugString());
    }
}
