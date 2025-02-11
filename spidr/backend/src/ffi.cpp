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
#include <cstdlib>
#include <cstddef>
#include <string>
#include <cstring>

#include "ffi.h"

extern "C" {
    void* deref(void** ptr) {
        return *ptr;
    }

    int isnull(void* ptr) {
        return ptr == nullptr;
    }

    string* string_new() {
        return reinterpret_cast<string*>(new std::string());
    }

    void string_delete(string* s) {
        delete reinterpret_cast<std::string*>(s);
    }

    char* string_c_str(string* s) {
        auto str = reinterpret_cast<std::string*>(s);
        auto len = str->size();
        auto res = (char *) malloc(len + 1);
        std::memcpy(res, str->c_str(), len * sizeof(char));
        return res;
    }

    char* string_data(string* s) {
        auto str = reinterpret_cast<std::string*>(s);
        auto len = str->size();
        auto res = (char *) malloc(len);
        std::memcpy(res, str->data(), len * sizeof(char));
        return res;
    }

    size_t string_size(string* s) {
        return reinterpret_cast<std::string*>(s)->size();
    }

    int sizeof_int() {
        return sizeof(int);
    }

    int sizeof_int64_t() {
        return sizeof(int64_t);
    }

    int sizeof_ptr() {
        return sizeof(void*);
    }

    // index() is a deprecated POSIX function that was silently
    // interfering with that name
    void* idx(int idx, void** ptr) {
        return ptr[idx];
    }

    void set_array_int(int* arr, int idx, int value) {
        arr[idx] = value;
    }

    void set_array_int64_t(int64_t* arr, size_t idx, int64_t value) {
        arr[idx] = value;
    }

    void set_array_ptr(void** arr, int idx, void* value) {
        arr[idx] = value;
    }
}
