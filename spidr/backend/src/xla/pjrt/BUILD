cc_library(
    name = "pjrt",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt:pjrt_executable",
        "//src",
        "//src/xla/client",
    ],
    visibility = ["//visibility:public"],
)
