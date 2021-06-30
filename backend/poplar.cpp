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
/* This file contains the C-compatible API to Poplar's C++ API. */
class Scalar {
    public:
        Scalar(double x) :xx{x} {};
        Scalar* add(Scalar* other) const {
            return new Scalar(this->xx + other->xx);
        };
        double toDouble() const {
            return this->xx;
        };

    private:
        double xx;
};

extern "C"
{
    struct cScalar;

    struct cScalar* cScalar_new(double x) {
        return reinterpret_cast<cScalar*>(new Scalar(x));
    }

    void cScalar_del(struct cScalar* s) {
        delete reinterpret_cast<Scalar*>(s);
    }

    struct cScalar* cScalar_add(struct cScalar* s, struct cScalar* other) {
        Scalar* s_ = reinterpret_cast<Scalar*>(s);
        Scalar* other_ = reinterpret_cast<Scalar*>(other);
        return reinterpret_cast<cScalar*>(s_->add(other_));
    }

    double cScalar_toDouble(struct cScalar* s) {
        return reinterpret_cast<Scalar*>(s)->toDouble();
    }
}
