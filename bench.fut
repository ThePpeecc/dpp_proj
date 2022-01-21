import "radix-4-way"
import "radix-8-way"
import "lib/github.com/diku-dk/sorts/radix_sort"
import "radix-intragroup"


-- ==
-- entry: test_radix_4_way test_radix_8_way test_radix_native test_radix_intragroup
-- random input { [10]u32 }
-- random input { [100]u32 }
-- random input { [1000]u32 }
-- random input { [10000]u32 }
-- random input { [100000]u32 }
-- random input { [1000000]u32 }
-- random input { [10000000]u32 }
-- random input { [100000000]u32 }

entry test_radix_4_way [n] (xs : [n]u32) = radix_sort_4_way xs

entry test_radix_8_way [n] (xs : [n]u32) = radix_sort_8_way xs

entry test_radix_intragroup [n] (xs : [n]u32) = radix_sort_intragroup xs

entry test_radix_native [n] (xs : [n]u32) = radix_sort 32 u32.get_bit xs 




-- Functions for testing different situtations for the algorithms
entry make_data_with_max_nums n k = map (%k) (iota n) |> map (u32.i64)

entry make_sorted_data n = reverse (iota n)




