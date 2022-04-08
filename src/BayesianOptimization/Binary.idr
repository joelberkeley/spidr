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
module BayesianOptimization.Binary

||| Wraps a binary, or arity two, function.
public export
record Binary a b c where
  constructor MkBinary
  run : a -> b -> c

export
Functor (Binary a b) where
  map f (MkBinary g) = MkBinary (f .: g)

export
Applicative (Binary a b) where
  pure x = MkBinary $ \_, _ => x
  (MkBinary f) <*> (MkBinary x) = MkBinary $ \a, b => f a b (x a b)

export
Monad (Binary a b) where
  join (MkBinary f) = MkBinary (\a, b => run (f a b) a b)

infixr 9 <|, |>, <|>

||| Apply a function to the left-most value of the binary function, before passing it to the binary
||| function.
|||
||| ```idris
||| add : Binary Int Int Int
||| add = MkBinary (+)
|||
||| add_fst : Binary (Int, Int) Int Int
||| add_fst = fst <| add
|||
||| three : Int
||| three = run add_fst (2, -4) 1
||| ```
export
(<|) : (a -> aa) -> Binary aa b c -> Binary a b c
f <| bin = MkBinary (\a, b => run bin (f a) b)

||| Apply a function to the right-most value of the binary function, before passing it to the binary
||| function. For example,
|||
||| ```idris
||| add : Binary Int Int Int
||| add = MkBinary (+)
|||
||| add_fst : Binary Int (Int, Int) Int
||| add_fst = fst |> add
|||
||| three : Int
||| three = run add_fst 1 (2, -4)
||| ```
export
(|>) : (b -> bb) -> Binary a bb c -> Binary a b c
f |> bin = MkBinary (\a, b => run bin a (f b))

export
(<|>) : forall t . (forall a . t a -> a) -> Binary a b c -> Binary (t a) (t b) c
f <|> (MkBinary run) = MkBinary (\x, y => run (f x) (f y))

-- these compile for (Labelled a b) but I didn't think too much about whether to keep them
-- export
-- sourceL : forall t . (forall a, b . t a b -> a) -> Binary a b out -> Binary (t a a') (t b b') out

-- export
-- sourceR : forall t . (forall a, b . t a b -> b) -> Binary a b out -> Binary (t a a') (t b b') out
