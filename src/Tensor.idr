module Tensor

import Data.Vect

%default total

----------------------------- core definitions ----------------------------

public export
Shape : {rank: Nat} -> Type
Shape {rank} = Vect rank Nat

-- todo this type gives users a lot of flexibility, and i might end up paying for
--   it in code complexity
public export
Array : Shape -> Type -> Type
Array [] dtype = dtype
Array (d :: ds) dtype = Vect d (Array ds dtype)

public export
data Tensor : (shape: Shape) -> (dtype: Type) -> Type where
  MkTensor : Array shape dtype -> Tensor shape dtype

-- this would enable interchanging scalars and numbers, but we can add that
-- later when we know if we want it (we probably don't _need_ it)
-- Num dtype => Num (Tensor [] dtype) where

export
Show (Array shape dtype) => Show (Tensor shape dtype) where
  show (MkTensor x) = "Tensor " ++ show x

----------------------------- structural operations ----------------------------

export
index : (idx: Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype
index idx (MkTensor x) = MkTensor $ index idx x

zipWith : {shape : Shape {rank}} -> (a -> b -> c) -> Tensor shape a -> Tensor shape b -> Tensor shape c
zipWith f (MkTensor x) (MkTensor y) = MkTensor (zipWithArray f x y) where
  zipWithArray : {shape': Shape {rank=rank'}} -> (a -> b -> c) -> Array shape' a -> Array shape' b -> Array shape' c
  zipWithArray {rank'=Z} {shape'=[]} f x y = f x y
  zipWithArray {rank'=(S k)} {shape'=(d :: ds)} f x y = zipWith (zipWithArray f) x y

export
transpose : Tensor [m, n] dtype -> Tensor [n, m] dtype
transpose (MkTensor x) = MkTensor $ transpose x

export
cross : Tensor

-- why does this work with a and b but not other names?
-- see http://docs.idris-lang.org/en/latest/tutorial/interfaces.html#functors-and-applicatives
public export
Functor (Tensor shape) where
  map f (MkTensor x) = MkTensor (g x) where
    g : Array s a -> Array s b
    g {s = []} y = f y
    g {s = (_ :: _)} ys = map g ys

export
replicate : (over : Shape) -> Tensor shape dtype -> Tensor (over ++ shape) dtype
replicate over (MkTensor x) = MkTensor (f over x) where
  f : (over: Shape) -> Array s dtype -> Array (over ++ s) dtype
  f [] x' = x'
  f (d :: ds) x' = replicate d (f ds x')

export
cast_dtype : Cast dtype dtype' => Tensor shape dtype -> Tensor shape dtype'
cast_dtype tensor = map cast tensor

export
foldr1 : (dtype -> dtype -> dtype) -> {leading : Shape {rank = axis}}
  -> (axis : Nat) -> Tensor (leading ++ _ :: tail) dtype -> Tensor (leading ++ tail) dtype

----------------------------- numeric operations ----------------------------

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

export partial
(@@) : Num dtype => Tensor [m, S n] dtype -> Tensor (S n :: ps) dtype -> Tensor (m :: ps) dtype
(@@) (MkTensor x) (MkTensor y) = MkTensor $ matmul' x y where
  partial
  matmul' : Num ty => Array [m', S n] ty -> Array (S n :: ps) ty -> Array (m' :: ps) ty
  matmul' [] _ = []
  matmul' (x :: xs) y = (dot x y) :: (matmul' xs y) where
    partial
    dot : Num ty => Vect (S n) ty -> Array (S n :: ps) ty -> Array ps ty
    dot {ps = []} x y = sum $ zipWith (*) x y
    dot {ps = (_ :: [])} x y = foldr1 (zipWith (+)) $ zipWith (\x_elem, y_row => map (* x_elem) y_row) x y
    dot {ps = (_ :: _ :: [])} x y = ?r

public export implicit doubleToScalar : Double -> Tensor [] Double
doubleToScalar = MkTensor

export
(+) : Num dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) t1 t2 = zipWith (+) t1 t2

export
(-) : Neg dtype => Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) t1 t2 = zipWith (-) t1 t2

export
(*) : Num dtype => Tensor [] dtype -> Tensor shape dtype -> Tensor shape dtype
(*) (MkTensor x) t = map (* x) t

||| floating point division. we don't support integer division
export
(/) : Fractional dtype => Tensor shape dtype -> Tensor [] dtype -> Tensor shape dtype
(/) t (MkTensor x) = map (/ x) t

export
exp_ew : Tensor shape Double -> Tensor shape Double
exp_ew x = map exp x

-- remember, we're only writing functions that exist on tensors. For example,
--  exp and pow are wrt to tensor multiplication, not elementwise

export partial
det : Neg dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
det {n = Z} (MkTensor [[a]]) = MkTensor a
det {n = S Z} (MkTensor [[a, b], [c, d]]) = MkTensor $ a * d - b * c
det {n = S (S Z)} (MkTensor [[a, b, c], [d, e, f], [g, h, i]]) = MkTensor $ a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

partial
adjugate : Neg dtype => Tensor [S n, S n] dtype -> Tensor [S n, S n] dtype
adjugate {n = Z} (MkTensor [[a]]) = ?r
adjugate {n = (S Z)} (MkTensor [[a, b], [c, d]]) = MkTensor [[d, -c], [-b, a]]
adjugate {n = (S (S Z))} (MkTensor [[a, b, c], [d, e, f], [g, h, i]]) = MkTensor [
    [e * i - f * h, - (d * i - f * g), d * h - e * g],
    [- (b * i - c * h), a * i - c * g, - (a * h - b * g)],
    [b * f - c * e, - (a * f - c * d), a * e - b * d]
  ]

export partial
inverse : Tensor [S n, S n] Double -> Maybe (Tensor [S n, S n] Double)
inverse x = let det_@(MkTensor d) = det x in if (d == 0.0) then Nothing else Just $ (adjugate x) / det_

-- todo should taking the diag or det return a dtype or a Tensor [] dtype?
--  does the signature of index tell us it must be Tensor [] dtype?
--  i guess it would be inconvenient to have to re-wrap it into a Tensor for
--  further calculations
export
diag : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
diag (MkTensor x) = MkTensor $ product $ diag x
