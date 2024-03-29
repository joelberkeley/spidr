cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:matrix",
        "@xla//xla/client/lib:prng",
        "//src/xla:xla",
        "//src/xla/client:client",
    ],
    visibility = ["//visibility:public"],
)
