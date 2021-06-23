#include "poplar.h"
#include <iostream>

int main() {
    cScalar_print(cScalar_add(cScalar_new(2.0), cScalar_new(3.0)));
    return 0;
}