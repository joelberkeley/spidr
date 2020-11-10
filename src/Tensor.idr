module Tensor

import Data.Vect

%default total

----------------------------- core definitions ----------------------------

-- todo we can include covariance if we make this `Vect rank Integer`
public export
Shape : {rank: Nat} -> Type
Shape {rank} = Vect rank Nat

-- todo this type gives users a lot of flexibility, and i might end up paying for
--   it in code complexity
public export
ArrayLike : Shape -> Type -> Type
ArrayLike [] dtype = dtype
ArrayLike (d :: ds) dtype = Vect d (ArrayLike ds dtype)

public export
data Tensor : (shape: Shape) -> (dtype: Type) -> Type where
  MkTensor : ArrayLike shape dtype -> Tensor shape dtype

export
Show (ArrayLike shape dtype) => Show (Tensor shape dtype) where
  show (MkTensor x) = "Tensor " ++ show x

----------------------------- structural operations ----------------------------

infixl 9 ++:  -- todo is this right?

public export
(++:) : {0 r, r' : Nat} -> Shape {rank=r} -> Shape {rank=r'} -> Shape {rank=r + r'}
(++:) [] y = y
(++:) (x :: xs) y = x :: (xs ++: y)

export
index : (idx: Fin d) -> Tensor (d :: ds) dtype -> Tensor ds dtype
index idx (MkTensor x) = MkTensor $ index idx x

-- todo if we can define flatmap, can we write this as map2 with do notation?
zipWith : {shape : _} -> (a -> b -> c) -> Tensor shape a -> Tensor shape b -> Tensor shape c
zipWith f (MkTensor x) (MkTensor y) = MkTensor (zipWithArray f x y) where
  zipWithArray : {shape': _} -> (a -> b -> c) -> ArrayLike shape' a -> ArrayLike shape' b -> ArrayLike shape' c
  zipWithArray {shape'=[]} f x y = f x y
  zipWithArray {shape'=(d :: ds)} f x y = zipWith (zipWithArray f) x y

export
transpose : {n, m : _} -> Tensor [m, n] dtype -> Tensor [n, m] dtype
transpose (MkTensor x) = MkTensor $ transpose x

-- why does this work with a and b but not other names?
-- see http://docs.idris-lang.org/en/latest/tutorial/interfaces.html#functors-and-applicatives
public export
{shape : _} -> Functor (Tensor shape) where
  map f (MkTensor x) = MkTensor (g x) where
    g : {s : _} -> ArrayLike s a -> ArrayLike s b
    g {s = []} y = f y
    g {s = (_ :: _)} ys = map g ys

export
replicate : (over : Shape) -> Tensor shape dtype -> Tensor (over ++ shape) dtype
replicate over (MkTensor x) = MkTensor (f over x) where
  f : (over: Shape) -> ArrayLike s dtype -> ArrayLike (over ++ s) dtype
  f [] x' = x'
  f (d :: ds) x' = replicate d (f ds x')

export
cast_dtype : Cast dtype dtype' => {shape : _} -> Tensor shape dtype -> Tensor shape dtype'
cast_dtype tensor = map cast tensor

-- export
-- foldr1 : {axis : _} -> (dtype -> dtype -> dtype) -> {leading : Shape {rank = axis}} -> (axis : Nat) -> Tensor (leading ++ _ :: tail) dtype -> Tensor (leading ++ tail) dtype

export
diag : (n : Nat) -> dtype -> Tensor [n, n] dtype

----------------------------- numeric operations ----------------------------

-- see https://www.python.org/dev/peps/pep-0465/#precedence-and-associativity
infixl 9 @@

export
(@@) : Num dtype => {0 leading : Shape} -> Tensor (leading ++: [S n]) dtype -> Tensor (S n :: ps) dtype -> Tensor (leading ++: ps) dtype

-- todo subsume vdot and matmul into @@
export
vdot : Num dtype => Tensor [m, S n] dtype -> Tensor [S n] dtype -> Tensor [m] dtype

export
matmul : Num dtype => Tensor [m, S n] dtype -> Tensor [S n, p] dtype -> Tensor [m, p] dtype

export
(+) : Num dtype => {shape : _} -> Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(+) t1 t2 = zipWith (+) t1 t2

export
(-) : Neg dtype => {shape : _} -> Tensor shape dtype -> Tensor shape dtype -> Tensor shape dtype
(-) t1 t2 = zipWith (-) t1 t2

export
(*) : Num dtype => {shape : _} -> Tensor [] dtype -> Tensor shape dtype -> Tensor shape dtype
(*) (MkTensor x) t = map (* x) t

||| floating point division. we don't support integer division
export
(/) : Fractional dtype => Tensor (leading ++: shape) dtype -> Tensor leading dtype -> Tensor (leading ++: shape) dtype
-- (/) t (MkTensor x) = map (/ x) t

---------------------------- other ----------------------------------

ew_eq : Eq dtype => Tensor shape dtype -> dtype -> Tensor shape Bool

any : Tensor shape Bool -> Bool

all : Tensor shape Bool -> Bool

export
det : Neg dtype => Tensor (leading ++: [S n, S n]) dtype -> Tensor leading dtype

adjugate : Neg dtype => Tensor (leading ++: [S n, S n]) dtype -> Tensor (leading ++: [S n, S n]) dtype

cholesky : Tensor [S n, S n] dtype => Maybe (Tensor [S n, S n] dtype)

export
inverse : {0 leading: _} -> Tensor (leading ++: [S n, S n]) Double -> Maybe $ Tensor (leading ++: [S n, S n]) Double
inverse x = let det_ = det x in if any (ew_eq det_ 0) then Nothing else Just $ (adjugate x) / det_

-- todo should taking the diag or det return a dtype or a Tensor [] dtype?
--  does the signature of index tell us it must be Tensor [] dtype?
--  i guess it would be inconvenient to have to re-wrap it into a Tensor for
--  further calculations
export
trace_product : Num dtype => Tensor [S n, S n] dtype -> Tensor [] dtype
trace_product (MkTensor x) = MkTensor $ product $ diag x
