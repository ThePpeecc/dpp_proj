let radix_sort_intragroup [n] (xs: [n]u32) : [n]u32 = 
  let block_size = 256
  let tdxs = iota (block_size)
  let bits = 4 
  let bit_used = 1 << bits -- 16
  let bit_mask = bit_used - 1
  let rest = block_size - ( n % block_size)
  let modXS = if n % block_size != 0 then xs ++ (replicate (rest) u32.highest) else xs
  -- Helper function begining

  let radix_sort_MM [m] (xs : [m]u32) (b : u32) : [m]u32 =
    let block_num = m / block_size
    -- Due to the fact that we have padded the input array
    let kernel_1 = map (\bid -> 
      let data  = xs[bid*block_size : (bid+1)*block_size] :> [block_size]u32
      let radix = map (\elem -> i32.u32 (elem >> b*bits & bit_mask)) data :> [block_size]i32  
      let masks = map (\i -> 
        map (\r -> if i == i64.i32 r then 1 else 0) radix
      ) (iota 16)
      let scannedMasks = map (scan (+) 0) masks -- This is a seg scan, but i'm lazy
      let block_d_sums = map last scannedMasks
      let merged_mask_scan = map2 (\r tdx -> if tdx > 0 then scannedMasks[r, tdx - 1] else 0) radix tdxs
      in ( merged_mask_scan, block_d_sums)
    ) (iota block_num)
    let prefixes = flatten (map (.0) kernel_1)
    let block_d_sums_permuted = map (.1) kernel_1
    let block_d_sums = flatten <| transpose block_d_sums_permuted
    let scanned_block_d_sums = [0] ++ scan (+) 0 block_d_sums -- Scan kernel
    let globals_pos = map (\idx -> 
      let data = xs[idx]
      let bdx  = idx / block_size
      let scanned_block_offset = (i64.u32 (data >> b*bits & bit_mask)) * block_num + bdx
      in scanned_block_d_sums[scanned_block_offset] + prefixes[idx]
    ) (iota m)
    let sorted = scatter (copy xs) globals_pos xs    
    in sorted
        
  let sortedModXS = loop modXS for i < 8 do radix_sort_MM modXS (u32.i32 i)
  in sortedModXS[:n]
  
-- ==
--  input {  [23u32, 45u32, 12u32, 34564u32, 2112u32, 3222u32, 123u32, 3211u32, 534u32, 2u32, 3u32, 112u32] }
--  output { [2u32, 3u32, 12u32, 23u32, 45u32, 112u32, 123u32, 534u32, 2112u32, 3211u32, 3222u32, 34564u32] }
let main [n] (xs : [n]u32) : [n]u32 = 
  radix_sort_intragroup xs
