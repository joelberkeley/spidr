cc_library(
    name = "builder",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/builder:xla_builder",
        "//src",
        "//src/xla",
    ],
    visibility = ["//visibility:public"],
)
