#include <iostream>
#include <unordered_map>

#include <torch/csrc/jit/api/module.h>

using namespace std;
using namespace torch::jit;

namespace fun {
class Compiler {
 private:
  unordered_map<string, Module> children = {};
  Module module;

 public:
  Compiler(Module module) : module(module) {
    for (const NameModule& s : module.named_children()) {
      std::cout << s.name << std::endl;
      children[s.name] = s.value;
    }
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
};
} // namespace fun
