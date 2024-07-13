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
# Design of a Bayesian Optimization Library

In this tutorial, we look at the design of spidr's Bayesian optimization functionality - functionality that uses historic data to optimize a black-box objective function.

## A Bayesian optimization refresher

Techniques such as stochastic gradient descent excel at optimizing functions whose value and gradient are cheap to evaluate. But when function evaluations are expensive and the gradient is unknown, these techniques can be infeasible. In Bayesian optimization we approach such problems by placing a probabilistic model over historical function evaluations, and optimizing over this surrogate model instead. Model evaluations are cheap, and the gradients are known. A probabilistic model captures the uncertainty in the objective function as a posterior _distribution_, rather than a point. Bayesian optimization algorithms can make use of this predictive distribution in their own way in order to find the objective optimum, for example choosing a particular balance of exploration and exploitation.

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

While we can trivially represent a number of new query points with a `Tensor`, we won't constrain ourselves to a particular representation for our data and models. We'll just call this representation our _environment_, and name it `env`. To find `n` new query points, we need a function `env -> Tag $ Tensor (n :: features) F64` (for continuous input space of features with shape `features`).

How we produce the new points from the data and models depends on the problem at hand. We could simply do a grid search over the mean of the model's marginal distribution for a single optimal point, as follows. We define some toy data

<!-- idris
import Control.Monad.Reader
import Control.Monad.Identity

import Data.Stream

import BayesianOptimization
import Data
import Distribution
import Model
import Model.GaussianProcess
import Model.Kernel
import Model.MeanFunction
import Optimize
import Tensor
-->
```idris
historicalData : Dataset [2] [1]
historicalData = MkDataset {
    features = tensor [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9]],
    targets = tensor [[1.2], [-0.5], [0.7]]
  }
```
and model that data
```idris
model : Dataset [2] [1] -> Tag $ ConjugateGPRegression [2]
model dataset =
  let mkGP = \len => pure $ MkGP zero (matern52 1.0 $ squeeze len)
      model = MkConjugateGPR mkGP (tensor [0.5]) 0.2
   in fit lbfgs dataset model
```
then optimize over the marginal mean
```idris
optimizer : Optimizer $ Tensor [1, 2] F64
optimizer f =
  let gs = gridSearch (tensor {a = Nat} [100, 100]) (tensor [0.0, 0.0]) (tensor [1.0, 1.0])
   in broadcast <$> (gs $ f . broadcast)

newPoint : Tag $ Tensor [1, 2] F64
newPoint = optimizer $ \x => squeeze <$>
  mean {event = [1]} !(marginalise @{Latent} !(model historicalData) x)
```

This is a particularly simple example of the standard approach of defining an _acquisition function_ over the input space which quantifies how useful it would be to evaluate the objective at a set of points, then finding the points that optimize this acquisition function. We can visualize this:

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

In this case, our acquisition function is built from the model and data (it is empirical). The optimizer is not empirical. Finally, the new points are empirical since they depend on the acquisition function. We can see from this simple setup that we want to be able to combine empirical objects and non-empirical objects to empirically find a new point. That is, we want to have a number of `env -> Tag a`: functions from data and models in a representation `env` to a number of `a` (the `Tag` simply allows us to efficiently reuse tensors when doing this), where the form of these functions depends on how we want to approach the problem at hand. We also want to be able to combine these `a` with non-empirical functionality.

## Modifying empirical values with `Functor`

In the above example, we constructed the acquisition function from our model, then optimized it, and in doing so, we assumed that we have access to the environment when we compose the acquisition function with the optimizer. This might not be the case: we may want to compose things before we get the data and model. For example, we may want to apply an `Optimizer` directly to an `env -> Tag $ Acquisition batch feat`. We want to be able to treat the data and model as an environment, and calculate and manipulate values in that environment. That's exactly what a _reader_ type does, and there's one in the Idris standard library, named `ReaderT`. A `ReaderT env Tag a` is just a thin wrapper round an `env -> Tag a`. Having chosen `ReaderT` as our abstraction, we want to apply an `Optimizer` to a `ReaderT env Tag (Acquisition batch feat)`. The function `map` from the `Functor` interface does just this, and `Functor` is already implemented for `ReaderT env Tag`. Let's see this in action:
```idris
modelMean : ProbabilisticModel [2] [1] Gaussian m => m -> Acquisition 1 [2]
modelMean model x = squeeze <$> mean {event = [1]} !(marginalise model x)

newPoint' : Tag $ Tensor [1, 2] F64
newPoint' = let acquisition = asks {m = ReaderT _ _} $ modelMean @{Latent}
                point = do lift $ optimizer !acquisition
             in runReaderT !(model historicalData) point
```

## Combining empirical values with `Applicative`

Let's now explore the problem of optimization with failure regions. We'll want to modify a measure `oa` of how optimal each point is likely to be (based on the objective value data), with a measure `fa` of how likely the point is to lie within a failure region (based on the failure region data). Both `oa` and `fa` are empirical values.

Combining empirical values will be a common pattern in Bayesian optimization. The standard way to do this with `ReaderT` values is with the two methods of the `Applicative` interface. The first of these lifts function application to the `ReaderT env Tag` context. For example, we can apply the `a -> b` function in `f : ReaderT env Tag (a -> b)` to the `a` value in `x : ReaderT env Tag a` as `f <*> x` (which is a `ReaderT env Tag b`), and we can do this before we actually have access to the environment. The second method, `pure`, creates a `ReaderT env Tag a` from any `a`.

There are a number of ways to implement the solution, but we'll choose a relatively simple one that demonstrates the approach, namely the case `fa : ReaderT env Tag (Acquisition batch feat)` and `oa : ReaderT env Tag (Acquisition batch feat -> Acquisition batch feat)`. We can visualise this:

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

## Specify the environment with contravariant functors

The `ReaderT env Tag a` type has proven flexible in allowing us to construct an acquisition tactic. Let's now look at how to construct our environment. spidr provides a minimal `DataModel` record that wraps a `Dataset` and `ProbabilisticModel`, and uses this as a common environment for building aquisition functions. But sometimes we'll want to use a different structure, and without adding complexity to the empiric values themselves. Recall that a `ReaderT env Tag a` is equivalent to a `env -> Tag a`, and that we can modify the `a` with a `Functor`. A `Functor` is really a _covariant functor_, and there's an "opposite" construct, called a _contravariant functor_ which has a similar effect on the function input. Idris has a `Contravariant` interface, but due to language limitations it's not suitable for `ReaderT`, so spidr provides a standalone function `(>$<)`, which fulfills the roles of `Contravariant`'s equivalent to `map`.

With this new functionality at hand, we'll return to our objective with failure regions. We'll need some data on failure regions, and to model that data:
```idris
failureData : Dataset [2] [1]
failureData = MkDataset {
    features = tensor [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9], [0.7, 0.1]],
    targets = tensor [[0.0], [0.0], [0.0], [1.0]]
  }

failureModel : Dataset [2] [1] -> Tag $ ConjugateGPRegression [2]
failureModel dataset
  = let mkGP = \len => pure $ MkGP zero (rbf $ squeeze len)
        model = MkConjugateGPR mkGP (tensor [0.2]) 0.1
     in fit lbfgs dataset model
```
We'll gather all the data and models in a `record`:
```idris
record Labelled o f where
  constructor Label
  objective : o
  failure : f
```
Idris generates two methods `objective` and `failure` from this `record`, which we'll use to extract the respective data and model. Putting it all together, here's our empirical point:
```idris
newPoint'' : Tag $ Tensor [1, 2] F64
newPoint'' = do
  let eci = objective >$< expectedConstrainedImprovement @{Latent} 0.5
      pof = failure >$< probabilityOfFeasibility @{%search} @{Latent} 0.5
      acquisition = do lift $ optimizer !(eci <*> pof)
  (hist, fail) <- tag (historicalData, failureData)
  let dataAndModel = Label (MkDataModel !(model hist) hist) (MkDataModel !(failureModel fail) fail)
  runReaderT dataAndModel acquisition
```

## Iterative Bayesian optimization with infinite data types

Once we've chosen some new points, we'll typically evaluate the objective function, which will look something like
```idris
objective : Tensor [n, 2] F64 -> Tag $ Tensor [n, 1] F64
```
and then update the historic dataset with this new point and train the model the new data. spidr provides, for simple Bayesian optimization setups, a function `step` which combines this all into a single step that we can reuse
```idris
step' : DataModel {probabilisticModel = Latent} (ConjugateGPRegression [2]) ->
        Tag $ DataModel {probabilisticModel = Latent} (ConjugateGPRegression [2])
step' = let tactic = do lift $ optimizer !(expectedImprovementByModel @{Latent})
         in step @{Latent} objective (fit lbfgs) tactic
```
We can repeat this process indefinitely to produce an infinite stream of values
```idris
covering
steps : Tag $ TagStream $ DataModel {probabilisticModel = Latent} (ConjugateGPRegression [2])
steps = tag historicalData >>= \hist => iterate step' (MkDataModel !(model hist) hist)
```
We can now iterate over this stream, choosing to stop according to a variety of stopping conditions, such as a number of repetitions
```idris
covering
firstFive : Tag $ Vect 5 (DataModel {probabilisticModel = Latent} $ ConjugateGPRegression [2])
firstFive = take 5 !steps
```
or a more complex stopping condition such when a new point lies close to a known optimum.

<!-- idris
main : IO ()
-->
