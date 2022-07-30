<!--
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
-->
# XLA: Accelerated Linear Algebra

## Why XLA?

We had the option to implement `Tensor` in pure Idris, or to build on a third party tool. We chose the latter for a number of reasons:

* it allowed us to start working on higher-level aspects of spidr sooner, such as the probabilistic modelling API
* many frameworks have been highly optimized and offer impressive performance
* many frameworks offer acceleration on hardware beyond CPUs

We were drawn to smaller third party tools that offered only what we needed and nothing more, and especially those that were newer as they would be more likely to have learnt from older frameworks. The first candidate was Graphcore's Poplar. While the speed of Graphcore's IPU was attractive, we ruled this out because IPUs are difficult to access for individuals, and other accelerators are either not emphaised or not supported at all. XLA was the next candidate. It supports a number of accelerators of interest (including the IPU), and it is currently being used by Google's JAX, which suggests it will remain active for the forseeable future. It also offers a C++ API which allows us to efficiently call into it from Idris. In retrospect, progress has been slower due to the fact that XLA does not include automatic differentiation. We're unsure if this would have affected our decisions had we considered this at the time.

## The foreign function interface to C++

As mentioned, XLA has a C++ API. In order to call this from Idris, we had two options. The first is to write a C++ backend for Idris. Apparently the Idris core language is small, which means writing new backends is less work than one may expect. The other option is to wrap XLA in a pure C wrapper and use Idris' FFI capabilities to call into this wrapper. We chose this second option for a number of reasons:

* we had more familiarity with how to FFI into C than we did in writing a backend
* all Idris backends support C FFI, which means that spidr would be compatible with all Idris backends
* the C wrapper may be useful for other projects as many languages support FFI to C

We decided to keep the C wrapper as close to the C++ interface as possible, within the limitations of what would work with Idris' FFI. We did this so that the C wrapper made few assumptions about how it would be used, and also so that we could implement as much of the logic as possible within Idris. Naturally, to do this we need a layer in Idris of very XLA-specific code that looks only approximately like spidr's tensor API. Thus, spidr is structured as
```
    Idris tensor API ------> XLA-specific Idris layer ------> C wrapper for XLA ------> XLA C++ interface
```

Let's take a look at these, starting from C++ and going left along the diagram to Idris. There are a number of important differences between C++ and C, and we must wrap C++ so that it can be consumed as a pure C API.

## Calling C++ from C

### C++ overloading and extern

C++ supports overloading, so functions names are always not enough to determine which function is being referred to. As such, names are mangled in such as way as to differentiate overload variants. C doesn't do this, and so it doesn't know how to use the mangled names. The solution is to give each overload a unique name and wrap all C++ code in `extern "C" { <C++ code goes here> }`. This tells the compiler not to mangle names.

### C++ classes

C++ is object-oriented. It has objects and classes. C does not. Let's look at how we can make a method on a C++ class compatible with C. Suppose we have the following simple C++ class
```cpp
class Foo {
  public:
    double Bar(int x);
}
```
we can start by making the class explicit in the function signature
```cpp
extern "C" {
  double Bar(Foo* foo, int x) {
    return foo->Bar(x);
  }
}
```
This is good, as we've hidden the method resolution machinery, but we still have a reference to the class `Foo`, and C doesn't use classes. We could resolve this by simply typing `foo` as a `void*`, but we can be clearer and safer by creating an _opaque pointer_ `c__Foo`, and casting to and from that, as
```cpp
extern "C" {
  struct c__Foo;

  double Bar(c__Foo* foo, int x) {
    Foo* foo_ = reinterpret_cast<Foo*>(foo);
    return foo_->Bar(x);
  }
}
```
This API is now pure C.

## Calling C from Idris

The basics of calling C from Idris are relatively trivial (see [the docs](https://idris2.readthedocs.io/en/latest/ffi/index.html)), but there are a number of things to watch out for when working with numeric values, memory management, side effects, and data structures.

### Primitive conversions

I FEEL OUT OF MY DEPTH ON THIS SECTION. NEED HELP

On the face of it, passing primitive types between Idris and C is easy. Say we want to call a C function
```c
int foo(int x);
```
we can write an Idris foreign function
<!-- idris
import Data.List
import System.FFI
-->
```idris
%foreign "C:foo,libfoo"
foo : Int -> Int
```
add call `foo 2`, `foo (-1)` no problem. All's good, right? Not quite. Integers in C and Idris can have different allowed ranges. Say your C compiler has a maximum `int` of 32767, what happens if we call `foo 40000`? Well, it's undefined. We can work around this by wrapping our Idris function and using fixed width numeric types
```idris
foo' : Int32 -> Int32
foo' = cast . foo . cast
```

### Memory management

### Side effects

### Data structures

Data structures in C are often represented as `struct`s or pointers. In Idris, you can pass both of these through FFI, and there are helper functions for getting and setting C `struct` members. However, data structures represented using pointers do not have the same level of support. For example, there are no functions for passing an Idris list to C. A C list is represented in Idris by a pointer to the first list element. To create this list, one must first allocate memory in C, then traverse the Idris list, setting the elements in the C list one by one:
```c
size_t sizeof_double () {
  return size_t sizeof(double);
}

void set(double* arr, unsigned int idx, double value) {
  arr[idx] = value;
}
```
```idris
%foreign "C:sizeof_double,libarray"
sizeof_double : Bits64

%foreign "C:set,libarray"
prim__setElem : Ptr Double -> Double -> Bits64 -> PrimIO ()

toClist : List Double -> IO (Ptr Double)
toClist xs = do
  clist <- malloc (cast (length xs) * cast sizeof_double)

  let clist = prim__castPtr clist

      setElem : Double -> Nat -> IO ()
      setElem elem idx = primIO (prim__setElem clist elem (cast idx))

  sequence_ (zipWith setElem xs [0..length xs])
  pure clist
```
Of course, one also needs to free the `Ptr Double` at some point, either manually or via garbage collection. A similar approach can also be used to build an Idris list from a C list:
```c
void get(double* arr, unsigned int idx) {
  return arr[idx]
}
```
```idris
%foreign "C:get,libarray"
get : Ptr Double -> Bits64 -> Double

fromClist : Nat -> Ptr Double -> List Double
fromClist len ptr = map (get ptr . cast) [0..len]
```

## A total API to foreign functions

The subset of Idris types allowed in foreign functions provide very little type and memory safety. spidr's approach to this is to wrap foreign functions with a well-typed interface, which captures all the requirements for the XLA API. Such a well-typed wrapper gives us confidence that our Idris interface to XLA is total. In spidr, we choose to write these functions to align closely with the C++ API, so that it's clear where we have re-interpreted the API to be more idiomatic or convenient.
