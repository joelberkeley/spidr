/*
Copyright 2021 Joel Berkeley

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
/* This file glues the C API to the Poplar C++ API. */
#include "poplar.h"
#include "stub.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

cScalar* cScalar_new(float x) {
    return reinterpret_cast<cScalar*>(new Scalar(x));
}

void cScalar_del(cScalar* s) {
    delete reinterpret_cast<Scalar*>(s);
}

cScalar* cScalar_add(cScalar* s, cScalar* other) {
    Scalar* s_ = reinterpret_cast<Scalar*>(s);
    Scalar* other_ = reinterpret_cast<Scalar*>(other);
    return reinterpret_cast<cScalar*>(s_->add(other_));
}

void cScalar_print(cScalar* s) {
    return reinterpret_cast<Scalar*>(s)->print();
}

#ifdef __cplusplus
}
#endif