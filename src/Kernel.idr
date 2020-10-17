module Kernel

import Data.Vect
import Tensor

-- avoid manipulating the underlying array directly. This code is essentially
-- user code and shouldn't have knowledge of the underyling Tensor representation

public export
Kernel : (samples : Nat) -> (features : Shape) -> Type
-- Kernel features = {samples : Nat} -> Tensor (samples :: features) Double -> Tensor (samples :: features) Double -> Tensor [samples, samples] Double
Kernel samples features = Tensor (samples :: features) Double -> Tensor (samples :: features) Double -> Tensor [samples, samples] Double

radial_basis_function : Tensor [] Double -> Kernel samples [domain_dim]
radial_basis_function {samples} {domain_dim} variance x x' = let squares = the (Tensor [samples, domain_dim] Double) (map (pow 2.0) (x - x')) in
                                                                 exp_ew $ foldr1 {leading=[samples]} {tail=[]} (+) 1 $ squares / (2.0 * variance)
