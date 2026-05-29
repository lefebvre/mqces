# Google Benchmark microbenchmarks for mqces. Built when
# MQCES_ENABLE_BENCHMARKS=ON; each .cpp is a self-contained executable.

set(MQCES_BENCHMARKS
    bench_spatial_rank.cpp
    bench_inverse_rank.cpp
    bench_score.cpp
    bench_classify_end_to_end.cpp
    bench_parallel_scaling.cpp
    bench_solver_comparison.cpp
    bench_scaling_million_row.cpp
)

foreach(_src IN LISTS MQCES_BENCHMARKS)
    get_filename_component(_name "${_src}" NAME_WE)
    add_executable(${_name} ${_src})
    target_link_libraries(${_name}
        PRIVATE
            mqces::mqces
            mqces::warnings
            benchmark::benchmark_main
    )
endforeach()
