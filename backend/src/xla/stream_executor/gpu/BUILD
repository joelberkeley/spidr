cc_library(
    name = "gpu",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",
        "//src/xla:xla",
        "//src/xla/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
