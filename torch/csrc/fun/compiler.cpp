#include <iostream>
#include <regex>
#include <unordered_map>

#include <torch/csrc/jit/api/module.h>

using namespace std;
using namespace torch::jit;

namespace fun {

class Allocator {
 private:
  uint64_t dram_base = 0;
  const uint64_t dram_capacity = ((uint64_t)1) << 32;

 public:
  Allocator() {
    std::cout << "dram_capacity: " << dram_capacity << std::endl;
  }
  uint64_t dram_allocate(uint64_t n) {
    // std::cout << "dram_allocate: " << n << std::endl;
    uint64_t new_dram_base = dram_base + n;
    TORCH_CHECK(new_dram_base <= dram_capacity, "dram_capacity not enough");
    uint64_t dram_base_bak = dram_base;
    dram_base = new_dram_base;
    return dram_base_bak;
  }
};

struct Conv2dParameter {
  int in_channels;
  int out_channels;
  int kernel_size_x;
  int kernel_size_y;
  int stride_x;
  int stride_y;
  int dilation_x;
  int dilation_y;
  bool transposed;
};

struct Pool2dParameter {
  int kernel_size_x;
  int kernel_size_y;
};

struct NNKnifeResult {
  int of_chiplet;
  int Kp;
  int Yp;
  int Kc;
  int act_tile_hor;
  int act_tile_chl;
  int act_str_line;
  int act_str_chl;
  int act_tile_ver;
};

NNKnifeResult NNKnife() {
  NNKnifeResult result;
  result.of_chiplet = 4;
  result.Kc = 4;
  result.Kp = 1;
  result.Yp = 4;
  return result;
}

bool is_module(torch::jit::Node* node, string str) {
  TORCH_CHECK(
      node->kind() == prim::CallMethod,
      "Kind of node to be prim::CallMethod, but got ",
      string(node->kind().toUnqualString()));

  auto type = node->inputs()[0]->type()->cast<c10::ClassType>();
  if (type && type->name()) {
    static std::regex mangle_re("\\.___torch_mangle_\\d+");
    auto qualified_name =
        std::regex_replace(type->name()->qualifiedName(), mangle_re, "");
    return qualified_name == str;
  }

  return false;
}

class Compiler {
 private:
  unordered_map<string, Module> children = {};
  unordered_map<torch::jit::Value*, uint64_t> address = {};
  Module module;
  Allocator* allocator = new Allocator();

 public:
  Compiler(Module module) : module(module) {
    for (const NameModule& s : module.named_children()) {
      children[s.name] = s.value;
    }
  }

  Conv2dParameter parseConv2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node, "__torch__.torch.nn.modules.conv.ConvTranspose2d"),
        "node to be Conv2d");

    Conv2dParameter param;

    auto value = node->inputs()[1];
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes().value();
    param.in_channels = sizes[1];
    param.out_channels = shape(node->output())[1];

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    for (auto&& i : children[child_name].named_parameters(false)) {
      if (i.name == "weight") {
        param.kernel_size_x = i.value.sizes()[2];
        param.kernel_size_y = i.value.sizes()[3];
        break;
      }
    }

    auto child_graph = children[child_name].get_method("forward").graph();
    auto _convolution_node = child_graph->outputs()[0]->node();

    auto dilation_list = _convolution_node->inputs()[5]->node()->inputs();
    param.dilation_x = dilation_list[0]->node()->i(attr::value);
    param.dilation_y = dilation_list[1]->node()->i(attr::value);

    auto stride_list = _convolution_node->inputs()[3]->node()->inputs();
    param.stride_x = stride_list[0]->node()->i(attr::value);
    param.stride_y = stride_list[1]->node()->i(attr::value);

    param.transposed = _convolution_node->inputs()[6]->node()->i(attr::value);

    return param;
  }

  Pool2dParameter parsePool2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d"),
        "node to be MaxPool2d");

    Pool2dParameter param;

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = children[child_name].get_method("forward").graph();
    auto max_pool2d_node = child_graph->outputs()[0]->node();
    auto kernel_size_list = max_pool2d_node->inputs()[1]->node()->inputs();

    param.kernel_size_x = kernel_size_list[0]->node()->i(attr::value);
    param.kernel_size_y = kernel_size_list[1]->node()->i(attr::value);

    return param;
  }

  std::vector<int64_t> shape(torch::jit::Value* value) {
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes();
    if (sizes.has_value()) {
      return sizes.value();
    } else {
      auto node = value->node();
      TORCH_CHECK(
          node->kind() == prim::CallMethod,
          "Kind of node to be prim::CallMethod, but got ",
          string(node->kind().toUnqualString()));

      const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
      auto child_graph = children[child_name].get_method("forward").graph();
      auto children_output =
          child_graph->outputs()[0]->type()->cast<TensorType>();
      return children_output->sizes().concrete_sizes().value();
    }
  }

  void allocateValue(torch::jit::Value* value) {
    auto s = shape(value);
    int64_t n = 0;
    if (s.size() == 0)
      n = 0;
    else
      n = s[0];

    for (int i = 1; i < s.size(); i++) {
      n *= s[i];
    }

    address[value] = allocator->dram_allocate(n);
  }

  void allocateNode(torch::jit::Node* node) {
    if (node->kind() == prim::GetAttr)
      return;

    auto outputs = node->outputs();
    for (auto&& i : outputs) {
      allocateValue(i);
    }
  }

  void allocateConv2dWeight(torch::jit::Value* value, Conv2dParameter param) {
    address[value] = allocator->dram_allocate(
        param.kernel_size_x * param.kernel_size_y * param.in_channels *
        param.out_channels);
  }

  void allocateActivation() {
    auto nodes = module.get_method("forward").graph()->nodes();
    for (auto&& node : nodes) {
      allocateNode(node);
    }
  }

  void printAddress() {
    for (const auto& n : address) {
      std::cout << "Value:[" << n.first->debugName() << "] Address:["
                << n.second << "]\n";
    }
  }

  void node_backend(torch::jit::Node*& node) {
    auto kind = node->kind();
    if (kind == prim::GetAttr) {
      return;
    }

    if (kind == aten::relu) {
      std::cout << "relu_en 1" << std::endl;
      std::cout << "relu_mode 00" << std::endl;
      std::cout << "relu_param 0_32" << std::endl;

      return;
    }

    if (kind == aten::leaky_relu) {
      std::cout << "relu_en 1" << std::endl;
      std::cout << "relu_mode 10" << std::endl;
      std::cout << "relu_param 0.01_32" << std::endl;

      return;
    }

    if (kind == aten::tanh) {
      std::cout << "relu_en 1" << std::endl;
      std::cout << "relu_mode 11" << std::endl;
      std::cout << "relu_param 0_32" << std::endl;

      return;
    }

    if (kind == prim::CallMethod) {
      auto GetAttrValue = node->inputs()[0];
      auto type = GetAttrValue->type()->cast<c10::ClassType>();
      TORCH_CHECK(type && type->name());

      static std::regex mangle_re("\\.___torch_mangle_\\d+");
      auto qualified_name =
          std::regex_replace(type->name()->qualifiedName(), mangle_re, "");

      if (qualified_name == "__torch__.torch.nn.modules.conv.Conv2d" ||
          qualified_name == "__torch__.torch.nn.modules.conv.ConvTranspose2d") {
        auto param = parseConv2d(node);
        allocateConv2dWeight(GetAttrValue, param);

        auto knifeResult = NNKnife();

        std::cout << "conv_type ";
        if (param.transposed)
          std::cout << "10";
        else {
          if (param.dilation_x == 1 && param.dilation_y == 1)
            std::cout << "00";
          else
            std::cout << "01";
        }
        std::cout << std::endl;

        std::cout << "Chiplet_mode ";
        if (knifeResult.Kp == 1 && knifeResult.Yp == 4)
          std::cout << "00";
        else if (knifeResult.Kp == 4 && knifeResult.Yp == 1)
          std::cout << "10";
        std::cout << std::endl;

        std::cout << "Kernel_num "
                  << param.kernel_size_x * param.kernel_size_y - 1 << std::endl;
        std::cout << "Kernel_width " << param.kernel_size_x - 1 << std::endl;
        std::cout << "Kernel_height " << param.kernel_size_y - 1 << std::endl;

        std::cout << "Weight_bit " << 0 << std::endl;
        std::cout << "weight_updata_n " << 0 << std::endl;
        std::cout << "act_tile_str " << 0 << std::endl; // todo: the first Conv2d should be 1

        std::cout << "Kernel_str ";
        if (param.stride_x == 1 && param.stride_y == 1)
          std::cout << "00";
        else if (param.stride_x == 2 && param.stride_y == 2)
          std::cout << "01";
        std::cout << std::endl;

        return;
      }

      if (qualified_name == "__torch__.torch.nn.modules.pooling.MaxPool2d") {
        std::cout << "Pooling_en 1" << std::endl;
        auto param = parsePool2d(node);
        auto size = param.kernel_size_x * param.kernel_size_y;
        std::cout << "pool_size " << size - 1 << std::endl;
        std::cout << "oprands " << 1.0 / size << std::endl;
        return;
      }

      std::cout << qualified_name << std::endl;

      TORCH_CHECK(false);
      return;
    }
  }

  void backend() {
    auto nodes = module.get_method("forward").graph()->nodes();
    for (auto&& node : nodes) {
      node_backend(node);
    }
  }
};
} // namespace fun
