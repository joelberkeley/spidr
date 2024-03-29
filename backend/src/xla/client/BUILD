cc_library(
    name = "client",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla/client:client_library",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/service:computation_placer",
        "@xla//xla/client:global_data",
        "@xla//xla/client:xla_builder",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "//src:src",
        "//src/xla:xla",
        "//src/xla/service:service",
    ],
    visibility = ["//visibility:public"],
)
