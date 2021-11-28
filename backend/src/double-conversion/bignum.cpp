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
/* This file contains the pure C API to XLA. */
#include <double-conversion/bignum.h>

using namespace double_conversion;

extern "C"
{
    struct c__Bignum;

    struct c__Bignum* c__Bignum_Bignum() {
        return reinterpret_cast<c__Bignum*>(new Bignum());
    }

    void c__Bignum_del(struct c__Bignum* s) {
        delete reinterpret_cast<Bignum*>(s);
    }

    void c__Bignum_AssignUInt64(struct c__Bignum& s, uint64_t x) {
        reinterpret_cast<Bignum&>(s).AssignUInt64(x);
    }

    void c__Bignum_AddBignum(struct c__Bignum& s, struct c__Bignum& other) {
        reinterpret_cast<Bignum&>(s).AddBignum(reinterpret_cast<Bignum&>(other));
    }

    int c__Bignum_Compare(struct c__Bignum& s, struct c__Bignum& other) {
        return Bignum::Compare(
            reinterpret_cast<Bignum&>(s),
            reinterpret_cast<Bignum&>(other)
        );
    }
}
