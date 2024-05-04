cc_library(
    name = "xla",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = ["@xla_extension//:xla_extension"],
    visibility = ["//visibility:public"],
)
