cc_library(
    name = "client",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client:executable_build_options",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "//src",
        "//src/xla",
    ],
    visibility = ["//visibility:public"],
)
