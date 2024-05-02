cc_library(
    name = "pjrt",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt:pjrt_executable",
        "//src"
    ],
    visibility = ["//visibility:public"],
)
