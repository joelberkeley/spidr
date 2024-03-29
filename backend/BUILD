cc_binary(  # see https://stackoverflow.com/a/51689102/5986907
    name = "c_xla",
    linkshared = True,
    srcs = [
        "//src/xla:xla",
        "//src/xla/client:client",
        "//src/xla/client/lib:lib",
        "//src/xla/pjrt:pjrt",
        "//src/xla/pjrt/c:pjrt_c",
        "//src/xla/service:service",
        "//src:src",
    ],
)
