module Optimize

import Tensor
import Data.Vect

public export
Optimizer : Type -> Type
Optimizer a = (a -> Maybe $ Tensor [] Double) -> Maybe a

export
grid_search : (density : Tensor [d] Double) -> (lower : Tensor [d] Double) -> (upper : Tensor [d] Double)
  -> (Tensor [] Double -> Tensor [] Double) -> Tensor [] Double
grid_search density lower upper f = ?grid_search_rhs
