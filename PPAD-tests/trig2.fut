import "scan-adj-comp"

-------------------------------------------------
---  4. Scan with 3x3 Matrix Multipplication  ---
-------------------------------------------------



def mm3by3  (a1: f32, b1: f32, c1: f32, d1: f32, e1: f32, f1: f32, g1: f32, h1: f32, i1: f32)
            (a2: f32, b2: f32, c2: f32, d2: f32, e2: f32, f2: f32, g2: f32, h2: f32, i2: f32) =

  ( 1*b1*c1*d1*e1*f1*g1*h1*i1*a2,
    a1*1*c1*d1*e1*f1*g1*h1*i1*b2,
    a1*b1*1*d1*e1*f1*g1*h1*i1*c2,
    a1*b1*c1*1*e1*f1*g1*h1*i1*d2,
    a1*b1*c1*d1*1*f1*g1*h1*i1*e2,
    a1*b1*c1*d1*e1*1*g1*h1*i1*f2,
    a1*b1*c1*d1*e1*f1*1*h1*i1*g2,
    a1*b1*c1*d1*e1*f1*g1*1*i1*h2,
    a1*b1*c1*d1*e1*f1*g1*h1*1*i2)

let mm3_ne = (1f32,1f32,1f32, 1f32,1f32,1f32, 1f32,1f32,1f32)

def primal3 [n] (xs: [n](f32,f32,f32,f32,f32,f32,f32,f32,f32)) =
  scan mm3by3 mm3_ne xs

def fromarrs3 = map (\(x: [9]f32) -> (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))
def toarrs3 = map (\(a,b,c,d,e,f,g,h,i) -> [a,b,c,d,e,f,g,h,i])

def onehot_2d n m x y =
  tabulate_2d n m (\i j -> f32.bool((i,j) == (x,y)))

entry fwd [n] (input: [n][9]f32) : [n][9][n][9]f32 =
  let input = fromarrs3 input
  in tabulate (n*9) (\i -> jvp primal3 input (fromarrs3 (onehot_2d n 9 (i/9) (i%9))))
     |> map toarrs3 |> transpose |> map transpose |> map (map unflatten)

entry rev [n] (input: [n][9]f32) : [n][9][n][9]f32 =
  let input = fromarrs3 input
  in tabulate (n*9) (\i -> vjp primal3 input (fromarrs3 (onehot_2d n 9 (i/9) (i%9))))
     |> unflatten |> map (map toarrs3)

-- Scan with 3x3 matrix multiplication: performance
-- ==
-- entry: scan_mm3_ours scan_mm3_prim scan_mm3_comp
-- compiled random input { [9][100000]f32      [9][100000]f32 }
-- compiled random input { [9][1000000]f32     [9][1000000]f32 }
-- compiled random input { [9][10000000]f32    [9][10000000]f32 }
-- compiled random input { [9][100000000]f32   [9][100000000]f32 }
-- compiled random input { [9][1000000000]f32  [9][1000000000]f32 }

def fromarrs3T [n] (x: [9][n]f32) = 
  map (\i -> (x[0,i],x[1,i],x[2,i],x[3,i],x[4,i],x[5,i],x[6,i],x[7,i],x[8,i])) (iota n)

entry scan_mm3_prim [n] (inp : [9][n]f32) 
                        (_adj: [9][n]f32) =
  fromarrs3T inp |> primal3

let zero = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32)
let plus (a1: f32, a2: f32, a3: f32, a4: f32, a5: f32, a6: f32, a7: f32, a8: f32, a9: f32)
         (b1: f32, b2: f32, b3: f32, b4: f32, b5: f32, b6: f32, b7: f32, b8: f32, b9: f32) =
  (a1+b1, a2+b2, a3+b3, a4+b4, a5+b5, a6+b6, a7+b7, a8+b8, a9+b9)

entry scan_mm3_comp [n] (inp : [9][n]f32)
                        (adj : [9][n]f32) =
  scan_bar zero plus mm3by3 mm3_ne
            (fromarrs3T inp)
            (fromarrs3T adj)

entry scan_mm3_ours [n] (inp : [9][n]f32)
                        (adj : [9][n]f32) =
  vjp primal3 (fromarrs3T inp) (fromarrs3T adj)