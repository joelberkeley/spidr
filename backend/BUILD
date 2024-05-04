cc_binary(
    name = "c_xla_extension",
    linkshared = True,
    srcs = [
        "//src/tensorflow/core/common_runtime/gpu:gpu",
        "//src/tensorflow/core/platform:platform",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/compiler/xla/client:client",
        "//src/tensorflow/compiler/xla/client/lib:lib",
        "//src/tensorflow/compiler/xla/service:service",
        "//src/tensorflow/stream_executor:stream_executor",
        "//src:src",
    ],
    deps = [
        "//src/tensorflow/core/common_runtime/gpu:gpu",
        "//src/tensorflow/core/platform:platform",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/compiler/xla/client:client",
        "//src/tensorflow/compiler/xla/client/lib:lib",
        "//src/tensorflow/compiler/xla/service:service",
        "//src/tensorflow/stream_executor:stream_executor",
        "//src:src",
    ],
)
