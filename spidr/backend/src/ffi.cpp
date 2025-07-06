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

    void string_delete(string* s) {
        delete reinterpret_cast<std::string*>(s);
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

    void set_array_ptr(void** arr, int idx, void* value) {
        arr[idx] = value;
    }
}
