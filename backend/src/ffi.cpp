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
extern "C" {
    void* deref(void** ptr) {
        return *ptr;
    }

    int isnull(void* ptr) {
        return ptr == nullptr;
    }

    void String_delete(String* s) {
        delete reinterpret_cast<std::string*>(s);
    }

    char* String_c_str(String* s) {
        auto str = reinterpret_cast<std::string*>(s);
        auto len = str.length();
        auto res = (char *) malloc(len);
        strncpy(res, str.c_str(), len);
        return res;
    }

    size_t String_size(String* s) {
        return reinterpret_cast<std::string*>(s).size();
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
}
