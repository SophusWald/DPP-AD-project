import "scan-adj-comp"

------------------------------------------
---  2. Scan with Lin-Fun Composition  ---
------------------------------------------

def plus_tup (a0: f32, a1: f32, a2: f32) (b0: f32, b1: f32, b2: f32) = (a0+a1, a1+b1, a2+b2)
def zero_tup = (0f32, 0f32, 0f32)


-- function composition (lfc2) operator;
let lfc (a0: f32, a1: f32, a2: f32)
        (b0: f32, b1: f32, b2: f32) : (f32, f32, f32) =
    ((f32.cosh a0)*b0 + b1*a0, a1*b1, f32.sinh (a2 + b2*a1 + b0*a0))

-- neutral element for linear-function composition
let lfc_ne = (0f32, 1f32, 0f32)

def primal_lfc [n] (xs: [n](f32,f32,f32)) =
  scan lfc lfc_ne xs

entry rev_Jlfc_ours [n] (a: [n]f32) (b: [n]f32) (c: [n]f32)=
  tabulate n (\i -> vjp primal_lfc (zip3 a b c) (replicate n (0,0,0) with [i] = (1,1,1)))
  |> map unzip3 |> unzip3

entry rev_Jlfc_comp [n] (a: [n]f32) (b: [n]f32) (c: [n]f32)=
  tabulate n (\i -> scan_bar zero_tup plus_tup lfc lfc_ne (zip3 a b c) 
                             (replicate n (0,0,0) with [i] = (1,1,1))
             )
  |> map unzip3 |> unzip3

-- Scan with linear-function composition: performance
-- ==
-- entry: scan_lfc_comp scan_lfc_ours scan_lfc_prim
-- compiled random input { [10000000]f32 [10000000]f32 [10000000]f32  [10000000]f32  [10000000]f32  [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32}

entry scan_lfc_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (_adj1 : [n]f32)
                        (_adj2 : [n]f32)
                        (_adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
  zip3 inp1 inp2 inp3 |> primal_lfc |> unzip3

entry scan_lfc_comp [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
  scan_bar zero_tup plus_tup lfc lfc_ne (zip3 inp1 inp2 inp3) (zip3 adj1 adj2 adj3)
    |> unzip3

entry scan_lfc_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
                        -- ([n]f32,[n]f32,[n]f32,[n]f32) =
  vjp primal_lfc (zip3 inp1 inp2 inp3) (zip3 adj1 adj2 adj3) |> unzip3
