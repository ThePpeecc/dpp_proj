import "radix-sort-mm"
import "radix-nn"
import "lib/github.com/diku-dk/sorts/radix_sort"


-- ==
-- entry: test_radix_nn test_radix_native
-- random input { [10]u32 }
-- random input { [100]u32 }
-- random input { [1000]u32 }
-- random input { [10000]u32 }
-- random input { [100000]u32 }
-- random input { [1000000]u32 }
-- random input { [10000000]u32 }
-- random input { [100000000]u32 }
-- script input { makedata 10i64 1000i64  }
-- script input { makedata 100i64 1000i64  }
-- script input { makedata 1000i64 1000i64  }
-- script input { makedata 10000i64 1000i64  }
-- script input { makedata 100000i64 1000i64  }
entry test_radix_mm [n] (xs : [n]u32) = radix_sort_mm xs


entry test_radix_nn [n] (xs : [n]u32) = radix_sort_nn xs


entry test_radix_native [n] (xs : [n]u32) = radix_sort 32 u32.get_bit xs 

entry makedata n k = map (%k) (iota n) |> map (u32.i64)




