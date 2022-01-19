
let segScan [n] 't (op: t -> t -> t) (ne:t) (arr: [n](t, bool)): [n]t =
  let modOp (v1: t, f1:bool ) (v2: t, f2: bool) = 
    (if f2 then v2 else op v1 v2, f1 || f2)
  in map (.0) (scan modOp (ne, false) arr)   

let sgmscan 't [n] (op: t->t->t) (ne: t) (flg : [n]i64) (arr : [n]t) : [n]t =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 != 0 then (f, x2)
            else (f, op x1 x2) )
         (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals


let radix_sort_step_nn [n] (xs : [n]u32) (b : u32) (bits : u32) : [n]u32 =
  let bits_len = i64.u32 (1<<bits)
  let get_bits x = x >> (b*bits) & ((1<<bits)-1)
  let pairwise op arr1 arr2 = map2 (op) arr1 arr2
  let bins = map get_bits xs
  let fflagsT = tabulate (n*bits_len) (\idx -> 
    let i = idx / bits_len
    let j = idx % bits_len
    in
    if i64.u32 bins[i] == j then 1 else 0 
  )
  let flagsForFlags = tabulate (n*bits_len) (\idx -> if idx % bits_len == 0 then 1 else 0) 
  let foffsets = sgmscan (+) 0 flagsForFlags fflagsT
  let offsets = transpose <| unflatten bits_len n foffsets
  --let flags = tabulate_2d n bits_len (\i j -> 
  --                               if i64.u32 bins[i] == j then 1
  --                               else 0)
  --let offsets = scan (pairwise (+)) (replicate bits_len 0) flags

  let get_last xs = last xs

  let offs = concat [0] (scan (+) 0 (get_last offsets))[:bits_len-1]
   
  let f bin arr = offs[i32.u32 bin]+arr[i32.u32 bin]-1
  let is = map2 f bins offsets
  in scatter (copy xs) is xs





let radix_sort_nn_7_way [n] (xs : [n]u32) : [n]u32 = 

  let mask = (1<<8)-1

  let get_num count num = (count >> 8u64*num) & mask
    
  let bucket_width = 256 -- 1024 does improve performance by roughly 10.000 ms

  let rest = bucket_width - (n % bucket_width)
  let modXS = xs ++ (replicate rest u32.highest)

  let buckets = tabulate (rest+n) (\idx -> if idx % bucket_width == 0 then 1i64  else 0)

  -- Step nn begin
  let radix_sort_step_nn_7_way [n] (xs : [n]u32) (b : u32) : [n]u32 =
    let bits_len = 1<<3
    let get_bits x = x >> (b*3) & 7
    let bins = map get_bits xs -- First get bits for input (a)
    let implicit = map (\x -> 1u64<<(8u64*(u64.u32 x))) bins -- Implicit representation (b)
    let prefix = sgmscan (+) 0 (buckets :> [n]i64) implicit -- local prefix sums (c) // Counts can be caclulated now 
    let num_buckets = n / bucket_width
    let pre_count = map2 (\pref b -> (get_num pref b) - 1) prefix (map u64.u32 bins) -- (e)
    let radix_count = tabulate (num_buckets*(i64.u32 bits_len)) (\idx ->  -- (h)
        let count = prefix[idx % num_buckets * bucket_width + bucket_width - 1]
        let num = u64.i64 (idx / num_buckets)
        in get_num count num
      )
    let global_offs = concat [0] (scan (+) 0 radix_count)[:num_buckets*(i64.u32 bits_len)-1] -- (i)
    let is = tabulate n (\idx -> -- (j)
      let bucket_num = idx / bucket_width
      let our_num    = i64.u32 bins[idx]
      let local_off = pre_count[idx]
      let global_off = global_offs[our_num * num_buckets + bucket_num]
      in i64.u64 (local_off+global_off)
    )

    in scatter (copy xs) is xs
  -- Step nn end

  let sortedModXS = loop modXS for i < 11 do radix_sort_step_nn_7_way modXS (u32.i32 i)
  in sortedModXS[:n]














let radix_sort_step_nn_fast_big [n] (xs : [n]u32) (b : i32) : [n]u32 =
  let bits = map (u32.get_bit b) xs   -- [1,0,1,0,1,0]

  let temp = scan (\s b -> s + (b<<15)+(1-b)) 0 bits
  let adBit = (1<<15)-1

  let offs = temp[n-1] & adBit

  let idxs = map2 (\b d -> if b == 1 
                               then i64.i32 ((d>>15)-1+offs)
                               else i64.i32 ((d&adBit)-1)) bits temp

  in scatter (copy xs) idxs xs -- sorted



let radix_sort_nn [n] (xs : [n]u32) : [n]u32 =
  --if n <= 100000 then 
  --  if n <= 1000 then
  --    if n <= 100 then
  --      loop xs for i < 8 do radix_sort_step_nn xs (u32.i32 i) 4
  --    else
  --      loop xs for i < 11 do radix_sort_step_nn xs (u32.i32 i) 3
  --  else 
  --     loop xs for i < 16 do radix_sort_step_nn xs (u32.i32 i) 2
  --else 
  radix_sort_nn_7_way xs
