<!--
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
-->
# Design: Bayesian optimization

## A Bayesian optimization refresher

In this tutorial on design in Bayesian optimization, we'll look at how we can use historic data on a black-box objective function to predict what inputs are likely to optimize a black-box function. We'll detail the architecture of spidr's Bayesian optimization functionality as we go.

Our task is to find a new point at which to evaluate our objective function, using historic function evaluations and a model or models of that data. We can represent this visually:

<pre>
+-----------------------+
|                       |
|  All data and models  |
|                       |
+-----------------------+
            |
            |
            v
     +-------------+
     |             |
     |  New point  |
     |             |
     +-------------+
</pre>

While we can trivially represent a number of new query points with a `Tensor`, we won't constrain ourselves to a particular representation for our data and models. We'll just name this representation `i` (for "in"). Thus, to find `n` new query points, we need a function `i -> Tensor (n :: features) Double` (for continuous input space of features with shape `features`).

How we produce the new points from the data and models depends on the problem at hand. We could simply do a grid search over the mean of the model's marginal distribution for a single optimal point, as follows. We define some toy data

<!-- idris
import Tensor
import BayesianOptimization
import Model
import Distribution
import Optimize
import Util
-->
```idris
historicData : Data {samples=3} [2] [1]
historicData = (const [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9]], const [[1.2], [-0.5], [0.7]])
```

and model that data

```idris
model : Either SingularMatrixError $ ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}
model = let prior = MkGP zero (rbf $ const 0.3)
            likelihood = MkGaussian (fill 0) (diag 0.22)
            (qp, obs) = historicData
         in map ?marginalise $ posterior prior likelihood (qp, squeeze obs)
```

then optimize over the marginal mean

```idris
optimizer : Optimizer $ Tensor [1, 2] Double
optimizer = let gs = gridSearch (const [100, 100]) (const [0.0, 0.0]) (const [1.0, 1.0])
             in \f => broadcast . gs $ f . broadcast

newPoint : Either SingularMatrixError $ Tensor [1, 2] Double
newPoint = pure $ optimizer $ squeeze . mean {event=[1]} . !model
```

This is a particularly simple example of the standard approach of defining an _acquisition function_ over the input space which quantifies how useful it would be evaluate the objective at a set of points, then finding the points that optimize this acquisition function. We can visualise this:

<pre>
+-----------------------+
|                       |
|  All data and models  |
|                       |
+-----------------------+
            |
            |
            v
    +---------------+
    |               |
    |  Acquisition  |
    |    function   |
    |               |
    +---------------+    +-------------+
            |            |             |
            |-----<------|  Optimizer  |
            v            |             |
    +--------------+     +-------------+
    |              |
    |  New points  |
    |              |
    +--------------+
</pre>

In this case, our acquisition function depends on the model (which in turn depends on the data). It is empirical. The optimizer does not depend on the data. Finally, the new points are empirical since they depend on the acquisition function. We can see from this simple setup that we want to be able to combine empirical objects and non-empirical objects to empirically find a new point. That is, we want to have a number of `i -> o`: functions from data and models in a representation `i` to a number of `o`, where the form of these functions depends on how we want to approach the problem at hand. We also want to be able to combine these `o` with non-empirical functionality.

## Modifying empirical values with `Functor`

In the above example, we constructed the acquisition function from our model, then optimized it, and in doing so, we assumed that we have access to the data and models when we compose the acquisition function with the optimizer. This might not be the case: we may want to compose things before we get the data and model. Using spidr's names, we'd apply an `Optimizer` to an `i -> Acquisition`. We'd normally do this with `map`, a method on the `Functor` interface, but functions, including `i -> o`, don't implement `Functor` (indeed, in Idris, they can't). We can however, wrap an `i -> o` in the `Morphism i o` type (also called `i ~> o` with a tilde) which does implement `Functor`. We can `map` an `Optimizer` over a `i ~> Acquisition`, as follows:

```
modelMean : ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]} -> Acquisition 1 [2]
modelMean model = squeeze . mean {event=[1]} . model

newPoint' : Either SingularMatrixError $ Tensor [1, 2] Double
newPoint' = let acquisition = map optimizer (Mor modelMean)  -- `Mor` makes a `Morphism`
             in pure $ run acquisition !model  -- `run` turns a `Morphism` into a function
```

## Combining empirical values with `Applicative`

Let's now explore the problem of optimization with failure regions. We'll want to modify a measure `oa` of how optimal each point is likely to be (based on the objective value data), with a measure `fa` of how likely the point is to lie within a failure region (based on the failure region data). Both `oa` and `fa` are empirical values.

Combining empirical values will be a common pattern in Bayesian optimization. The standard way to do this with `Morphism` values is with the two methods of the `Applicative` interface. The first of these lifts function application to the `Morphism` context. For example, we can apply the `a -> b` function in `f : i ~> (a -> b)` to the `a` value in `x : i ~> a` as `f <*> x` (which is an `i ~> b`), and we can do this before we actually have access to any `i` values. The second method, `pure`, simply creates an `i ~> o` from a simple `o` value.

There are a number of ways to implement the solution, but we'll choose a relatively simple one that demonstrates the approach. Namely, the case `fa : i ~> Acquisition` and `oa : i ~> (Acquisition -> Acquisition)`. We can visualise this:

<pre>
+---------------------------------------+
|                                       |
|          All data and models          |
|                                       |
+---------------------------------------+
        |                    |
        |                    |
        v                    v
+---------------+    +------------------------------+
|               |    |                              |
|  Acquisition  |    |  Acquisition -> Acquisition  |
|    function   |    |    function       function   |
|               |    |                              |
+---------------+    +------------------------------+
        |                    |
        +---+----------------+
            |
            v
    +---------------+
    |               |
    |  Acquisition  |
    |    function   |
    |               |
    +---------------+    +-------------+
            |            |             |
            |------<-----|  Optimizer  |
            v            |             |
    +--------------+     +-------------+
    |              |
    |  New points  |
    |              |
    +--------------+
</pre>

The final point is then gathered from `map optimizer (oa <*> fa)`, and this concludes our discussion of the core design. Next, we'll implement this in full, and introduce some convenience syntax on the way.

## Separating representation from computation with `Empiric`

The `Morphism i o`, or `i ~> o` type, has proven flexible in allowing us to construct an acquisition tactic. However, since our representation of `i` of all the data and models is completely unconstrained, our `i ~> o` values will need to know how to handle this representation. Alongside actually constructing the empirical value, this means our `i ~> o` is doing two things. It would be nice to be able to separate these concerns of representation and computation. Taking a look at `modelMean`, while that only uses the model directly, other acquisition functions can also depend directly on the data. All acquisitions will depend on at least the model _or_ the data, and these two always appear together, we can define an atomic empirical value as one that takes one data set and the corresponding model. We call this an `Empiric`. We can then pair each `Empiric` value `emp` with functionality `f` to gather the data set and model from the `i` value, with the infix operator `>>>` as `f >>> emp`. This turns out to be a practical API for most cases, and where it doesn't fulfil our needs, we can always construct our `i ~> o` explicitly.

With this new functionality at hand, we'll return to our objective with failure regions. First we define some data on failure regions, and model that data

```
failureData : Data {samples=4} [2] [1]
failureData = (const [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9], [0.7, 0.1]], const [[0], [0], [0], [1]])

failureModel : Either SingularMatrixError $
               ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}
```

Then we choose a representation of our data and models. We'll use the following named pair

```
record Labelled o f where
  constructor Label
  objective : o
  failure : f
```

Idris generates two methods `objective` and `failure` from this `record`, which we'll use to get the respective data and model. Putting it all together, here's our empirical point:

```
newPoint'' : Either SingularMatrixError $ Tensor [1, 2] Double
newPoint'' = let eci = objective >>> expectedConstrainedImprovement
                 pof = failure >>> probabilityOfFeasibility (const 0.5)
                 acquisition = map optimizer (eci <*> pof)
                 dataAndModel = Label (historicData, !model) (failureData, !failureModel)
              in pure $ run acquisition dataAndModel
```
