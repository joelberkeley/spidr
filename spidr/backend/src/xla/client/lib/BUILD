cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client/lib:arithmetic",
        "@xla//xla/client/lib:constants",
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:matrix",
        "@xla//xla/client/lib:prng",
        "//src/xla",
        "//src/xla/client",
    ],
    visibility = ["//visibility:public"],
)
