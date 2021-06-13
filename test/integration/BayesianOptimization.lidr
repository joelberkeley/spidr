{--
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
--}
In this tutorial, we'll look how, in Bayesian optimization, we can use historic data to find a new point (or points) to query of some black-box objective function. We'll detail the architecture of spidr's Bayesian optimization functionality as we go.

When finding a new point, we have at our disposal our historic observations. In the simplest case, this will be a pair of input points and objective values. We'll also construct a model of how input points map to objective values. That is, we have some historic data and a model over that data.

In more complex cases, one data set and one model might be too restrictive. For example, if the objective fails to evaluate at some points, it might not be appropriate to model both the objective value (a regression problem) and its failure regions (a classification problem) together. Since our model of the objective values will only be over a subset of the data, one way we could represent this* is to have separate data sets for the objective value and failure regions data. More generally, we can expect to have any number of data sets, and one model for each data set.

Our task is to find a new point at which to evaluate our objective function, using our data and models. We can represent this visually:

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

While we can trivially represent a number of new query points with a `Tensor`, we won't constrain ourselves yet to a particular representation for our data and models, but we can name this representation `i` (for "in"). Thus, to find `n` new query points, we need a function `i -> Tensor (n :: features) Double`, where we've assumed a continuous input space of features of shape `features`.

How we produce the new point from the data and models depends on the problem at hand. We could simply do a grid search over the mean of the model's marginal distribution for the optimal point, as follows. First we define some data

< import Tensor
< import BayesianOptimization
< import Model
< import Distribution
< import Optimize
<
> historicData : Data {samples=3} [2] [1]
> historicData = (const [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9]], const [[1.2], [-0.5], [0.7]])

and model that data

> model : Maybe $ ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}
> model = let prior = MkGP zero linear
>             likelihood = MkGaussian ?mean ?cov
>             (qp, obs) = historicData
>          in map ?marginalise $ posterior prior likelihood (qp, squeeze obs)

then we optimize over the marginal mean

TODO gridSearch usage is messy
> optimizer : Optimizer $ Tensor [1, 2] Double
> optimizer = let gs = gridSearch (const [100, 100]) (const [0.0, 0.0]) (const [1.0, 1.0])
>              in \f => map broadcast $ gs $ f . broadcast
>
> newPoint : Maybe $ Tensor [1, 2] Double
> newPoint = optimizer $ Just . squeeze . mean {event_shape=[1]} . !model

This is a particularly simple example of the standard approach of defining an _acquisition function_ over the input space which quantifies how useful it would be evaluate the objective at a set of points, then finding the points that optimize this acquisition function. We can visualise this:

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

In this case, our acquisition function depends on the model (which in turn depends on the data). It is empirical. On the other hand, the optimizer does not. The new points are empirical since they depend on the acquisition function. We can see from this simple setup that we want to be able to combine empirical objects and non-empirical objects to empirically find a new point. That is, we want to have a number of `i -> o` from data and models in a representation `i`, which create a number of `o`, whose form depends on the problem at hand. We also want to be able to combine these `o` with non-empirical functionality.

Now when it comes to implementing empirical acquisition functions, if they are built from the representation `i`, they'll need some way of handling that representation. This means our empirical objects are doing more than one thing. We'd like to separate the task of handling the `i` from the task of constructing the `o`. So that empirical functionality doesn't need to think about representation, we can use an minimal subset of data and models. Specifically, everything empirical will use a data set, and every data set is paired with a model, so a reasonable option would be for each empirical object to use one data set and one model. Objects which need more than this can be constructed from multiple empirical objects. Note that we could have decided that each object use _either_ a data set _or_ a model, but that would mean we'd need two kinds of empirical object when, since we're doing Bayesian optimization, every data set will be paired with a model anyway and vice versa.

In spidr, these minimal empirical objects are `Empiric`s. An `Empiric features o` produces an empirical value of type `o` from a single data set and model (for features with shape `features`). We'll want to be able to modify the output of an `Empiric` (for example to optimize an empiricial acquisition function to get an empirical set of points) and, as we'll see in a moment, to combine the output of multiple `Empiric`s. To help us think about this, let's look at the example of a scalar objective with failure regions. We'll use an acquisition function which combines a measure of how optimal each point is likely to be (based on the objective value data), with a measure of how likely the point is to lie within a failure region (based on the failure region data). Since we're using two data sets, we'll need two `Empiric`s, where the output of one is combined with the output of the other to complete the acquisition function. Finally, the acquisition function must be optimized like in the previous example. The picture now looks as follows:

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

In order to combine each empirical value, we could construct each individually with the relevant data and model, then combine the results, but we might not have access to the data at the point we connect them together. Instead, we'll look at how to connect them together before we apply the data and models.

Since each `Empiric` takes one data set and model, we can't in general combine two `Empiric`s into a single `Empiric` because they may use different data and models. Thus, to combine `Empiric`s we must first bundle each one with a mechanism to get the specific data and model from the representation `i`. That is, we want one `i -> (Data, ProbabilisticModel)` for each `Empiric`. In spidr, we do this with the `Connection i o` type. A `Connection i o` is like an `Empiric` but it uses all the data and models and therefore a `Connection i o` and a `Connection i o'` only differ in what they produce.

It now makes sense to think about combining empirical values by combining `Connection i` values. How then do we do that? By making `Connection i` an applicative functor. What's that? Let's take a look in the context of the above diagram. If we'd constructed each empirical value first, then combined them, we'd have a `x : Acquisition 1 [2]` and an `f : Acquisition 1 [2] -> Acquisition 1 [2]`, from which we construct a complete acquisition function as `f x`. But we're not constructing our empirical values yet, so we have a `x_conn : Connection i Acquisition 1 [2]` and `f_conn : Connection i (Acquisition 1 [2] -> Acquisition 1 [2])`, so we need to apply the function in the context of a `Connection`. An applicative functors provides a method for exactly this: lifting function application to a context. In Idris, we write this `f_conn <*> x_conn`. Similarly, we can optimize the acquisition function in the resulting `y_conn : Connection i Acquisition` using `map`, as `map optimizer y_conn`, resulting in a `Connection i (Tensor 1 [2] Double)`. This is precisely the empirical points we're after.

OK TO HERE

Let's now implement this last example, and we'll look at a particular representations of data and models on the way. First off we'll need some data on failure regions. We'll reuse the data from above for objective values.

> failureData : Data {samples=4} [2] [1]
> failureData = (const [[0.3, 0.4], [0.5, 0.2], [0.3, 0.9], [0.7, 0.1]], const [[0], [0], [0], [1]])

We'll model the failure regions

> failureModel : Maybe $ ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}
> failureModel = ?failureModel_rhs

Now we'll choose a representation for all our data. We'll use a simple named pair

> record Labelled o f where
>   constructor Label
>   objective : o
>   failure : f

We can now construct our empirical point. We'll use `run` to convert our `Connection i o` to a function `i -> o` and apply it to the data and models.

> newPoint' : Maybe $ Tensor [1, 2] Double
> newPoint' = let eci = objective >>> expectedConstrainedImprovement {s=_}
>                 pof = failure >>> (probabilityOfFeasibility $ const 0.5) {s=_}
>                 acquisition = map optimizer (eci <*> pof)
>                 dataAndModel = Label (historicData, !model) (failureData, !failureModel)
>              in run acquisition dataAndModel
