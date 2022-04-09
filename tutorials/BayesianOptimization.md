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

In this tutorial, we look at the design of spidr's Bayesian optimization functionality - functionality that uses historic data to optimize a black-box objective function.

## A Bayesian optimization refresher

Techniques such as Adam excel at optimizing functions whose value and gradient are cheap to evaluate, but when function evaluations are expensive and the gradient is unknown, these techniques can be infeasible. In Bayesian optimization we approach such problems by placing a probabilistic model over historic function evaluations, and substituting objective function evaluations with model evaluations. Model evaluations are cheap, and the gradients are known, but unlike the objective function which produces exact values (neglecting noise), a probabilistic model produces a predictive distribution. We must therefore adopt some method to choose candidate optima from the predictive distribution. This mapping from distribution to a notion of optimality can balance exploration and exploitation in the search for the objective optima.

## Bayesian optimization design

We can represent choosing candidate optima visually:

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

While we can trivially represent a number of new query points with a `Tensor`, we won't constrain ourselves to a particular representation for our data and models. We'll just name this representation `i` (for "in"). Thus, to find `n` new query points, we need a function `i -> Tensor (n :: features) F64` (for continuous input space of features with shape `features`).

How we produce the new points from the data and models depends on the problem at hand. We could simply do a grid search over the mean of the model's marginal distribution for a single optimal point, as follows. We define some toy data

<!-- idris
import Literal
import Tensor
import BayesianOptimization
import Data
import Model
import Model.GaussianProcess
import Model.Kernel
import Model.MeanFunction
import Distribution
import Optimize
import Data.Stream
-->
```idris
historicData : Dataset [2] [1]
historicData = MkDataset (fromLiteral [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9]]) (fromLiteral [[1.2], [-0.5], [0.7]])
```

and model that data

```idris
model : ConjugateGPRegression [2]
model = let mk_gp = \len => MkGP zero (matern52 1.0 $ squeeze len)
            model = MkConjugateGPR mk_gp (fromLiteral [0.5]) 0.2
         in fit lbfgs historicData model
```

then optimize over the marginal mean

```idris
optimizer : Optimizer $ Tensor [1, 2] F64
optimizer = let gs = gridSearch (fromLiteral [100, 100]) (fromLiteral [0.0, 0.0]) (fromLiteral [1.0, 1.0])
             in \f => broadcast . gs $ f . broadcast

newPoint : Tensor [1, 2] F64
newPoint = optimizer $ squeeze . mean {event=[1]} . (marginalise @{Latent} model)
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

```idris
modelMean : ProbabilisticModel [2] [1] Gaussian m => m -> Acquisition 1 [2]
modelMean model = squeeze . mean {event=[1]} . (marginalise model)

newPoint' : Tensor [1, 2] F64
newPoint' = let acquisition = map optimizer $ MkBinary (\_ => modelMean @{Latent})
             in run acquisition historicData model  -- `run` turns a `Binary` into a function
```

## Combining empirical values with `Applicative`

Let's now explore the problem of optimization with failure regions. We'll want to modify a measure `oa` of how optimal each point is likely to be (based on the objective value data), with a measure `fa` of how likely the point is to lie within a failure region (based on the failure region data). Both `oa` and `fa` are empirical values.

Combining empirical values will be a common pattern in Bayesian optimization. The standard way to do this with `Morphism` values is with the two methods of the `Applicative` interface. The first of these lifts function application to the `Morphism` context. For example, we can apply the `a -> b` function in `f : i ~> (a -> b)` to the `a` value in `x : i ~> a` as `f <*> x` (which is an `i ~> b`), and we can do this before we actually have access to any `i` values. The second method, `pure`, creates an `i ~> o` from an `o`.

There are a number of ways to implement the solution, but we'll choose a relatively simple one that demonstrates the approach, namely the case `fa : i ~> Acquisition` and `oa : i ~> (Acquisition -> Acquisition)`. We can visualise this:

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

The `Morphism i o`, or `i ~> o` type has proven flexible in allowing us to construct an acquisition tactic. However, since our representation of `i` of all the data and models is completely unconstrained, our `i ~> o` values will need to know how to handle this representation. Alongside actually constructing the empirical value, this means our `i ~> o` is doing two things. It would be nice to be able to separate these concerns of representation and computation. Consider for example `modelMean`. While that only uses the model directly, other acquisition functions can also depend directly on the data. Everything empirical depends on at least a model or some data, and in Bayesian optimization these two always appear together. In spidr, we choose to define an atomic empirical value as one that takes any subset of the data, and the corresponding model of that subset of data. We call this an `Empiric`. We can then compose each `Empiric` `emp` with functionality `f` to gather the data set and model from the `i` value. We provide the infix operator `>>>` for this, used as `f >>> emp`. This turns out to be a practical API for most cases, and where it doesn't fulfil our needs, we can always construct our `i ~> o` explicitly.

With this new functionality at hand, we'll return to our objective with failure regions. We'll need some data on failure regions, and to model that data. Recall that we can represent this in any form we like, and we'll simply use a dedicated `Data` set and `ProbabilisticModel`:

```idris
failureData : Dataset [2] [1]
failureData = MkDataset (fromLiteral [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9], [0.7, 0.1]]) (fromLiteral [[0], [0], [0], [1]])

failureModel : ConjugateGPRegression [2]
failureModel = let mk_gp = \len => MkGP zero (rbf $ squeeze len)
                   model = MkConjugateGPR mk_gp (fromLiteral [0.2]) 0.1
                in fit lbfgs historicData model
```

and we'll gather all the data and models in a `record`:

```idris
record Labelled a where
  constructor Label
  objective : a
  failure : a
```

Idris generates two methods `objective` and `failure` from this `record`, which we'll use to extract the respective data and model. Putting it all together, here's our empirical point:

```idris
newPoint'' : Tensor [1, 2] F64
newPoint'' = let eci = objective >$< expectedConstrainedImprovement @{Latent} 0.5
                 pof = failure >$< probabilityOfFeasibility @{%search} @{Latent} 0.5
                 acquisition = map optimizer (eci <*> pof)
              in run acquisition (Label historicData failureData) (Label model failureModel)
```

## Iterative Bayesian optimization with infinite data types

Once we've chosen some new points, we'll typically evaluate the objective function, which will look something like

```idris
objective : Tensor [n, 2] F64 -> Dataset [2] [1]
```

at these points. We can then update our historical data and models with these new observations, in whatever way is appropriate for our chosen representation. Suppose we used a `Pair` of data and model, and collected one data point, this may look like

We can repeat the above process indefinitely, and spidr provides a function `loop` for this. It takes a tactic `i ~> Tensor (n :: features) F64` like we discussed in earlier sections, an observer as above, and initial data and models. Now we could have also asked the user for a number of repetitions after which it should stop, or a more complex stopping condition such when a new point lies within some margin of error of a known optimum. However, this would be unnecessary, and could make it harder to subsitute our stopping condition for another. Instead, we choose to separate the concern of stopping from the actual iteration. Without a stopping condition, `loop` thus must produce a potentially-infinite sequence of values. It can do this with the `Stream` type.

```idris
iterations : Stream (Dataset [2] [1], ConjugateGPRegression [2])
iterations = let tactic = map optimizer (expectedImprovementByModel @{Latent})
              in loop tactic objective (fit lbfgs) historicData model
```

We can peruse the values in this `Stream` in whatever way we like. We can simply take the first five iterations

```idris
firstFive : Vect 5 (Dataset [2] [1], ConjugateGPRegression [2])
firstFive = take 5 iterations
```

or use more complex stopping conditions as mentioned earlier. Unfortunately, we can't give an example of this because spidr lacks the functionality to define conditionals based on `Tensor` data.
