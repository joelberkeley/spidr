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
    int isnull(void* ptr);

    struct string;
    void string_delete(string* s);
    char* string_c_str(string* s);
    size_t string_size(string* s);

    int sizeof_int();
    int sizeof_ptr();

    size_t size(char*);
    void* index(int idx, void** ptr);
    void set_array_int(int* arr, int idx, int value);
    void set_array_ptr(void** arr, int idx, void* value);
}
