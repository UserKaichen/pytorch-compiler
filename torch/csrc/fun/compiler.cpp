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
    std::cout << "dram_allocate: " << n << std::endl;
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
  int kernel_size;
};

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
      std::cout << s.name << std::endl;
      children[s.name] = s.value;
    }
  }

  Conv2dParameter parseConv2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.conv.Conv2d"),
        "node to be Conv2d");

    Conv2dParameter param;

    auto value = node->inputs()[1];
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes().value();
    param.in_channels = sizes[1];
    param.out_channels = shape(node->output())[1];

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = children[child_name].get_method("forward").graph();
    for (auto&& i : children[child_name].named_parameters(false)) {
      if (i.name == "weight") {
        param.kernel_size = i.value.sizes()[2];
        break;
      }
    }

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

  void allocate() {
    auto nodes = module.get_method("forward").graph()->nodes();
    for (auto&& node : nodes) {
      allocateNode(node);
    }

    // Iterate and print keys and values of unordered_map
    for (const auto& n : address) {
      std::cout << "Key:[" << n.first->debugName() << "] Value:[" << n.second
                << "]\n";
    }
  }
};
} // namespace fun
