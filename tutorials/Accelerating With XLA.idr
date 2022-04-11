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
# Accelerating with XLA

When writing spidr's `Tensor` API, we had the option to implement `Tensor` within spidr, or use a third party tool. We chose to opt for a third party tool for a number of reasons:

* it allowed us to start working on higher-level aspects of spidr sooner, such as the probabilistic modelling API
* many frameworks have been highly optimized and offer impressive performance
* many frameworks offer acceleration on hardware beyond CPUs

We were drawn to smaller third party tools that offered only what we needed and nothing more, and especially those that were newer as they would be more likely to have learnt from older frameworks. The first candidate was Graphcore's Poplar. While the speed of Graphcore's IPU was attractive, we ruled this out because IPUs are difficult to access for individuals, and other accelerators are either not emphaised or not supported at all. XLA was the next candidate. It supports a number of accelerators of interest (including the IPU), and it is currently being used by Google's JAX, which suggests it will remain active for the forseeable future. It also offers a C++ API which allows us to efficiently call into it from Idris. In retrospect, progress has been slower due to the fact that XLA does not include automatic differentiation. We're unsure if this would have affected our decisions had we considered this at the time.

As mentioned, XLA has a C++ API. In order to call this from Idris, we had two options. The first is to write a C++ backend for Idris. Apparently the Idris core language is small, which means writing new backends is less work than one may expect. The other option is to wrap XLA in a pure C wrapper and use Idris' FFI capabilities to call into this wrapper. We chose this second option for a number of reasons:

* we had more familiarity with how to FFI into C than we did in writing a backend
* all Idris backends support C FFI, which means that spidr would be compatible with all Idris backends
* the C wrapper may be useful for other projects as many languages support FFI to C

We decided to keep the C wrapper as close to the C++ interface as possible, within the limitations of what would work with Idris' FFI. We did this so that the C wrapper made few assumptions about how it would be used, and also so that we could implement as much of the logic as possible within Idris. Naturally, to do this we need a layer in Idris of very XLA-specific code that looks only approximately like spidr's tensor API. Thus, spidr is structured as
```
    Idris tensor API ------> XLA-specific Idris layer ------> C wrapper for XLA ------> XLA C++ interface
```
