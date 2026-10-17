#include <torch/extension.h>

torch::Tensor leaky_relu_cpu(const torch::Tensor& x) {
    auto flat = x.reshape({-1});
    auto result = torch::empty_like(flat);

    auto input = flat.accessor<float, 1>();
    auto output = result.accessor<float, 1>();

    for (int64_t i = 0; i < flat.numel(); ++i) {
        output[i] = input[i] >= 0 ? input[i] : 0.1f * input[i];
    }

    return result.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_cpu);
}
