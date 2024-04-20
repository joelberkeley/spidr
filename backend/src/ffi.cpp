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
#include <iostream>

#include "ffi.h"

extern "C" {
    void* deref(void** ptr) {
        return *ptr;
    }

    int isnull(void* ptr) {
        return ptr == nullptr;
    }

    void string_delete(string* s) {
        delete reinterpret_cast<std::string*>(s);
    }

    char* string_data(string* s) {
//        // std::cout << "string_c_str ..." << std::endl;
        auto str = reinterpret_cast<std::string*>(s);
//        // std::cout << "... s" << std::endl;
//        // std::cout << *str << std::endl;
//        // std::cout << "... s length: " << str->length() << std::endl;
        auto len = str->size();
        auto res = (char *) malloc(len);
        std::memcpy(res, str->data(), len * sizeof(char));
//        // std::cout << "... res" << std::endl;
//        fwrite(res, sizeof(char), len, stdout);
//        // std::cout << std::endl;
        return res;
    }

    size_t string_size(string* s) {
        return reinterpret_cast<std::string*>(s)->size();
    }

    int sizeof_int() {
        return sizeof(int);
    }

    int sizeof_ptr() {
        return sizeof(void*);
    }

    void* index(int idx, void** ptr) {
        return ptr[idx];
    }

    void set_array_int(int* arr, int idx, int value) {
        arr[idx] = value;
    }

    void set_array_ptr(void** arr, int idx, void* value) {
        arr[idx] = value;
    }
}
