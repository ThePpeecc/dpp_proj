let sgmscan 't [n] (op: t->t->t) (ne: t) (flg : [n]i64) (arr : [n]t) : [n]t =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 != 0 then (f, x2)
            else (f, op x1 x2) )
         (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals


let radix_sort_4_way [n] (xs : [n]u32) : [n]u32 = 

  let mask = (1<<8)-1

  let get_num count num = (count >> 8u32*num) & mask
    
  let bucket_width = 256 -- 1024 does improve performance by roughly 10.000 ms

  let rest = bucket_width - (n % bucket_width)
  let modXS = xs ++ (replicate rest u32.highest)

  let buckets = tabulate (rest+n) (\idx -> if idx % bucket_width == 0 then 1  else 0)

  -- Step begin
  let radix_sort_step_4_way [n] (xs : [n]u32) (b : u32) : [n]u32 =
    let bits_len = 1<<2
    let get_bits x = x >> (b*2) & 3

    let bins = map get_bits xs -- First get bits for input (a)
    
    let implicit = map (\x -> 1u32<<(8u32*x)) bins -- Implicit representation (b)

    
    let prefix = sgmscan (+) 0 (buckets :> [n]i64) implicit -- local prefix sums (c) // Counts can be caclulated now 

    let num_buckets = n / bucket_width

    let pre_count = map2 (\pref b -> (get_num pref b) - 1) prefix bins -- (e)


    let radix_count = tabulate (num_buckets*(i64.u32 bits_len)) (\idx ->  -- (h)
        let count = prefix[idx % num_buckets * bucket_width + bucket_width - 1]
        let num = u32.i64 (idx / num_buckets)
        in get_num count num
      )
    
    let global_offs = concat [0] (scan (+) 0 radix_count)[:num_buckets*(i64.u32 bits_len)-1] -- (i)

    let is = tabulate n (\idx -> -- (j)
      let bucket_num = idx / bucket_width

      let our_num    = i64.u32 bins[idx]

      let local_off = pre_count[idx]
      
      
      let global_off = global_offs[our_num * num_buckets + bucket_num]
      in i64.u32 (local_off+global_off)
    )

    in scatter (copy xs) is xs
  -- Step end

  let sortedModXS = loop modXS for i < 16 do radix_sort_step_4_way modXS (u32.i32 i)
  in sortedModXS[:n]



let main [n] (xs: [n]u32) : [n]u32 = radix_sort_4_way xs
