cc_library(
    name = "xla",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla:literal",
    ],
    visibility = ["//visibility:public"],
)
