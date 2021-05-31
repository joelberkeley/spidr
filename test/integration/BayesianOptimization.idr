import Data.Vect
import Tensor
import Kernel
import MeanFunction
import GaussianProcess
import Distribution
import BayesianOptimization
import Optimize
import Util

historic_data : Data [2] [1]

public export 0 Model : Type
Model = ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}

model : Model

optimizer : AcquisitionOptimizer

new_point : Maybe $ Tensor [1, 2] Double
new_point = let ei = direct expectedImprovementByModel
                acquisition = map optimizer ei
             in apply acquisition (historic_data, model)

data Map : k -> v -> Type where

idx : k -> Map k v -> v

infixl 9 >>>

(>>>) : k -> (ty -> o) -> Connection (Map k ty) o
(>>>) key = MkConnection (idx key)

data_model_mapping : Map String $ Pair (Data [2] [1]) Model

new_point_constrained : Maybe $ Tensor [1, 2] Double
new_point_constrained = let eci = "OBJECTIVE" >>> expectedConstrainedImprovement
                            pof = "CONSTRAINT" >>> (probabilityOfFeasibility $ MkTensor 0.5)
                            acquisition = map optimizer $ eci <*> pof
                         in apply acquisition data_model_mapping
