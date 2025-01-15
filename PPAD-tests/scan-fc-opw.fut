import "scan-adj-comp"

------------------------------------------
---  2. Scan with Lin-Fun Composition  ---
------------------------------------------

def plus_tup (a0: f32, a1: f32, a2: f32) (b0: f32, b1: f32, b2: f32) = (a0+b0, a1+b1, a2+b2)
def zero_tup = (0f32, 0f32, 0f32)


-- function composition (fc) operator;
let fc (a0: f32, a1: f32, a2: f32)
        (b0: f32, b1: f32, b2: f32) : (f32, f32, f32) =
    (a0 ** (f32.log b0), a1 * b1, a2 + b2)

-- neutral element for linear-function composition
let fc_ne = (f32.exp 1, 1f32, 0f32)

def primal_fc [n] (xs: [n](f32,f32,f32)) =
  scan fc fc_ne xs

entry rev_Jfc_ours [n] (a: [n]f32) (b: [n]f32) (c: [n]f32)=
  tabulate n (\i -> vjp primal_fc (zip3 a b c) (replicate n (0,0,0) with [i] = (1,1,1)))
  |> map unzip3 |> unzip3

entry rev_Jfc_comp [n] (a: [n]f32) (b: [n]f32) (c: [n]f32)=
  tabulate n (\i -> scan_bar zero_tup plus_tup fc fc_ne (zip3 a b c) 
                             (replicate n (0,0,0) with [i] = (1,1,1))
             )
  |> map unzip3 |> unzip3

-- Scan with linear-function composition: performance
-- ==
-- entry: scan_fc_comp scan_fc_ours scan_fc_prim
-- compiled random input { [10000000]f32 [10000000]f32 [10000000]f32  [10000000]f32  [10000000]f32  [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32}

entry scan_fc_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (_adj1 : [n]f32)
                        (_adj2 : [n]f32)
                        (_adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
  zip3 inp1 inp2 inp3 |> primal_fc |> unzip3

entry scan_fc_comp [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
  scan_bar zero_tup plus_tup fc fc_ne (zip3 inp1 inp2 inp3) (zip3 adj1 adj2 adj3)
    |> unzip3

entry scan_fc_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (adj1 : [n]f32)
                        (adj2 : [n]f32)
                        (adj3 : [n]f32) : 
                        ([n]f32,[n]f32,[n]f32) =
                        -- ([n]f32,[n]f32,[n]f32,[n]f32) =
  vjp primal_fc (zip3 inp1 inp2 inp3) (zip3 adj1 adj2 adj3) |> unzip3
