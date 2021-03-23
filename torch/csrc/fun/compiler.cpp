#include <regex>
#include <iostream>
#include <unordered_map>

#include <torch/csrc/jit/api/module.h>
#include "./address.cpp"

using namespace std;
using namespace torch::jit;

namespace fun {
int num[4];
int layer_num;
int padding_num;
int layer_num_bf;
string layer_type;
string layer_type_bf;
bool BasicBlock_flag;
int  downsample_flag;

torch::jit::Node* node_back;
torch::jit::Node* relu_node_bk;
torch::jit::Value* GetAttrValue;

class Allocator {
 private:
  uint64_t dram_base = 0;
  const uint64_t dram_capacity = ((uint64_t)1) << 32;

 public:
  Allocator() {
  }
  uint64_t dram_allocate(uint64_t n) {
    uint64_t new_dram_base = dram_base + n;
    TORCH_CHECK(new_dram_base <= dram_capacity, "dram_capacity not enough");
    uint64_t dram_base_bak = dram_base;
    dram_base = new_dram_base;
    return dram_base_bak;
  }
};

struct Conv2dParameter {
  int stride_x;
  int stride_y;
  int dilation_x;
  int dilation_y;
  bool transposed;
  int in_channels;
  int out_channels;
  int kernel_size_x;
  int kernel_size_y;
  int feature_map_size_x;
  int feature_map_size_y;
};

struct BatchNormParameter {
  bool  flag;
  int   dimen;
};

struct ReluParameter {
  int en;
  string mode;
  string param;
};

struct AdaptParameter {
  int output_size_x;
  int output_size_y;

};

struct PaddingParameter {
  int dim_x;
  int dim_y;
  int dim_z;
  int dim_w;
};

struct LinearParameter {
  int in_features_x;
  int in_features_y;
  int out_features_x;
  int out_features_y;
};

struct Pool2dParameter {
  int stride_x;
  int stride_y;
  int kernel_size_x;
  int kernel_size_y;
};

struct NNKnifeResult {
  int of_chiplet;
  int Y2;
  int X2;
  int K2;
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
  result.Y2 = 2;
  result.X2 = 4;
  result.K2 = 1;
  result.Kc = 4;
  result.Kp = 1;
  result.Yp = 4;
  return result;
}

ReluParameter param_relu;
Conv2dParameter param_conv;
BatchNormParameter param_bn;

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
  int BasicBlock_cnt = 0;
  // Number of BasicBlocks that have been run
  string downsample_name;
  // The child name of the downsample
  string Sequential_name;
  // The child name of the Sequential
  string BasicBlock_name;
  // The child name of the BasicBlock
  string BasicBlock_total;
  // The total name of the BasicBlock

  unordered_map<string, Module> children = {};
  unordered_map<string, Module> block_children = {};
  unordered_map<string, Module> subblock_children = {};
  unordered_map<string, Module> Seqblock_children = {};
  unordered_map<string, Module> subSeqblock_children = {};
  unordered_map<string, Module> Seqsubblock_children = {};
  unordered_map<torch::jit::Value*, uint64_t> address = {};
  Module module;
  Allocator* allocator = new Allocator();

 public:
  Compiler(Module module) : module(module) {
    for (const NameModule& s : module.named_children()) {
      children[s.name] = s.value;
    }
  }

  BatchNormParameter parseBatchNorm(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.batchnorm.BatchNorm2d"),
        "node to be BatchNorm2d");

    BatchNormParameter param;
    if (BasicBlock_flag) {
      auto child_graph = module.get_method("forward").graph();
      const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
      if (downsample_flag == 2) {
        child_graph = Seqsubblock_children[child_name].get_method("forward").graph();
      } else {
        child_graph = subblock_children[child_name].get_method("forward").graph();
      }
      auto sizes = shape(child_graph->inputs()[1]);
      param.dimen = sizes[1];
    } else {
        auto sizes  = shape(node->inputs()[1]);
        param.dimen = sizes[1];
    }

     param.flag  = true;

    return param;
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

    TORCH_CHECK(
        node->kind() == prim::CallMethod,
        "Kind of node to be prim::CallMethod, but got ",
        string(node->kind().toUnqualString()));

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);

    auto child_graph = module.get_method("forward").graph();
    if (BasicBlock_flag) {
      // the conv layer in downsample, downsample in BasicBlock, BasicBlock in Sequential 
      if (downsample_flag == 2) {
        auto SeqsubBasicBlock_module = subblock_children[downsample_name];
        for (const NameModule& s : SeqsubBasicBlock_module.named_children()) {
          Seqsubblock_children[s.name] = s.value;
        }
        child_graph = Seqsubblock_children[child_name].get_method("forward").graph();
        for (auto&& i : Seqsubblock_children[child_name].named_parameters(false)) {
          if (i.name == "weight") {
            param.kernel_size_x = i.value.sizes()[2];
            param.kernel_size_y = i.value.sizes()[3];
            break;
          }
        }
      } else {
      /* the conv layer in BasicBlock */
        auto subBasicBlock_module = block_children[BasicBlock_name];
        for (const NameModule& s : subBasicBlock_module.named_children()) {
          subblock_children[s.name] = s.value;
        }
        child_graph = subblock_children[child_name].get_method("forward").graph();
        for (auto&& i : subblock_children[child_name].named_parameters(false)) {
          if (i.name == "weight") {
            param.kernel_size_x = i.value.sizes()[2];
            param.kernel_size_y = i.value.sizes()[3];
            break;
          }
        }
      }
      auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
      auto sizes = children_output->sizes().concrete_sizes().value();
      param.out_channels = sizes[1];
    }
    else {
    /* the conv layer in forward */
      child_graph = children[child_name].get_method("forward").graph();
      for (auto&& i : children[child_name].named_parameters(false)) {
        if (i.name == "weight") {
          param.kernel_size_x = i.value.sizes()[2];
          param.kernel_size_y = i.value.sizes()[3];
          break;
        }
      }
      param.out_channels = shape(node->output())[1];

    }

    auto size = pt->sizes().concrete_sizes(); 
    if (size.has_value()) {
      auto sizes = pt->sizes().concrete_sizes().value();
      param.in_channels = sizes[1];
    } else {
      auto sizes = shape(child_graph->inputs()[1]);
      param.in_channels = sizes[1];
    }

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

  PaddingParameter parseCat(torch::jit::Node* node) {
    PaddingParameter param;

    auto result_shape = shape(node->outputs()[0]);

    param.dim_x = result_shape[0];
    param.dim_y = result_shape[1];
    param.dim_z = result_shape[2];
    param.dim_w = result_shape[3];

    return param;
  }

  AdaptParameter parseAdapt(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d"),
        "node to be BatchNorm2d");

    AdaptParameter param;
    auto value = node->inputs()[1];

    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto size = pt->sizes().concrete_sizes();
    if (size.has_value()) {
        auto sizes = pt->sizes().concrete_sizes().value();
        param.output_size_x = sizes[2];
        param.output_size_y = sizes[3];
    } else {
        TORCH_CHECK(
                node->kind() == prim::CallMethod,
                "Kind of node to be prim::CallMethod, but got ",
                string(node->kind().toUnqualString()));
        const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
        auto child_graph = children[child_name].get_method("forward").graph();
        auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
        auto sizes = children_output->sizes().concrete_sizes().value();
        param.output_size_x = sizes[2];
        param.output_size_y = sizes[3];
    }

    return param;
  }

  Pool2dParameter parsePool2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d") ||
        is_module(node, "__torch__.torch.nn.modules.pooling.AvgPool2d"),
        "node to be MaxPool2d or AvgPool2d");

    Pool2dParameter param;

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = module.get_method("forward").graph();
    if (BasicBlock_flag) {
      child_graph = block_children[child_name].get_method("forward").graph();
    } else {
      child_graph = children[child_name].get_method("forward").graph();
    }
    auto pool2d_node = child_graph->outputs()[0]->node();
    auto kernel_size_list = pool2d_node->inputs()[1]->node()->inputs();

    param.kernel_size_x = kernel_size_list[0]->node()->i(attr::value);
    param.kernel_size_y = kernel_size_list[1]->node()->i(attr::value);

    auto stride_list = pool2d_node->inputs()[2]->node()->inputs();
    param.stride_x = stride_list[0]->node()->i(attr::value);
    param.stride_y = stride_list[1]->node()->i(attr::value);

    return param;
  }

  LinearParameter parseLinear(torch::jit::Node* node) {
    TORCH_CHECK(
      is_module(node, "__torch__.torch.nn.modules.linear.Linear"),
      "node to be Linear");

    LinearParameter param;

    auto value = node->inputs()[1];
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
  
    auto size = pt->sizes().concrete_sizes();
    if (size.has_value()) {
      auto sizes = pt->sizes().concrete_sizes().value();
      param.in_features_x  = sizes[0];
      param.in_features_y  = sizes[1];
      param.out_features_x = shape(node->output())[0];
      param.out_features_y = shape(node->output())[1];
    } else {
      unordered_map<string, Module> grand_children = {};
      auto Sequential_module = children["classifier"];
      for (const NameModule& s : Sequential_module.named_children()) {
        grand_children[s.name] = s.value;
      }
  
      const std::string& linear_name = node->inputs()[0]->node()->s(attr::name);
      auto grandchild_graph = grand_children[linear_name].get_method("forward").graph();
  
      auto children_output = grandchild_graph->outputs()[0]->type()->cast<TensorType>();
      auto sizes = children_output->sizes().concrete_sizes().value();
      param.out_features_x  = sizes[0];
      param.out_features_y  = sizes[1];
  
      auto s = shape(grandchild_graph->inputs()[1]);
      param.in_features_x = s[0];
      param.in_features_y = s[1];
    }
  
    return param;
  }

  void BasicBlock_node(torch::jit::Value* value) {
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes();
    auto node = value->node();
    TORCH_CHECK(
        node->kind() == prim::CallMethod,
        "Kind of node to be prim::CallMethod, but got ",
        string(node->kind().toUnqualString()));

    auto BasicBlock_module = children[Sequential_name];
    for (const NameModule& s : BasicBlock_module.named_children()) {
      block_children[s.name] = s.value;
      BasicBlock_total = s.name;
    }

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    BasicBlock_name = child_name;
    auto child_graph = block_children[child_name].get_method("forward").graph();
    auto nodes = child_graph->nodes();
    for (auto&& node : nodes) {
      node_backend(node, 0);
    }
  }

  void Sequential_node(torch::jit::Value* value) {
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);

    auto node = value->node();
    TORCH_CHECK(
        node->kind() == prim::CallMethod,
        "Kind of node to be prim::CallMethod, but got ",
        string(node->kind().toUnqualString()));

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = module.get_method("forward").graph();
    if (child_name == "downsample") {
      downsample_flag = 2;
      downsample_name = child_name;
      child_graph = subblock_children[child_name].get_method("forward").graph();
    } else if (child_name == "0" or child_name == "1") {
      //name 0 Sequential find name 0 BasicBlock
      auto subsequential_module = Seqblock_children[child_name];
      for (const NameModule& s : subsequential_module.named_children()) {
        subSeqblock_children[s.name] = s.value;
      }
      Sequential_name = child_name;
      child_graph = subSeqblock_children[child_name].get_method("forward").graph();
    } else {
      Sequential_name = child_name;
      child_graph = children[child_name].get_method("forward").graph();
      auto sequential_module = children[child_name];
      for (const NameModule& s : sequential_module.named_children()) {
        Seqblock_children[s.name] = s.value;
      }
    }

    auto nodes = child_graph->nodes();
    for (auto&& node : nodes) {
      node_backend(node, 0);
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
     auto child_graph = module.get_method("forward").graph();

     if (BasicBlock_flag == 1) {
       if (downsample_flag != 1) {
        auto subBasicBlock_module = block_children[BasicBlock_name];
        for (const NameModule& s : subBasicBlock_module.named_children()) {
          subblock_children[s.name] = s.value;
        }
       child_graph = subblock_children[child_name].get_method("forward").graph();
       } else if (downsample_flag == 1) {
        child_graph = Seqsubblock_children[child_name].get_method("forward").graph();
       }
     } else {
         child_graph = children[child_name].get_method("forward").graph();
     }
     auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
     return children_output->sizes().concrete_sizes().value();
   }
  }

  void allocateValue(torch::jit::Value* value) {
    auto node = value->node();
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
    auto kind = node->kind();
    if (kind == prim::GetAttr)
      return;
    else if (kind == aten::Int || kind == aten::size)
      return;
    else if (kind == prim::Constant || kind == prim::ListConstruct)
      return;
    else if ((kind != aten::zeros) || (kind != aten::relu) || \
            (kind != prim::NumToTensor) || (kind != aten::cat) \
             || (kind != aten::view)) {
      return;
    }
    else if (is_module(node, "__torch__.torch__.BasicBlock")) {
      return;
    }
    else if (is_module(node, "__torch__.torchvision.models.resnet.BasicBlock")) {
      return;
    }
    else if (is_module(node, "__torch__.torch.nn.modules.container.Sequential")) {
      return;
    }

    auto outputs = node->outputs();
    for (auto&& i : outputs) {
      allocateValue(i);
    }
  }

  uint64_t allocateConv2dWeight(
    torch::jit::Value* value,
    Conv2dParameter param) {
    auto weight_address = allocator->dram_allocate(
        param.kernel_size_x * param.kernel_size_y * param.in_channels *
        param.out_channels);
    address[value] = weight_address;
    return weight_address;
  }

  void allocateActivationAndInput() {
    auto graph = module.get_method("forward").graph();

    allocateValue(graph->inputs()[1]);
    auto nodes = graph->nodes();
    int i = 0;
    for (auto&& node : nodes) {
      i++;
      allocateNode(node);
    }
  }

  void printAddress() {
    for (const auto& n : address) {
      std::cout << "Value:[" << n.first->debugName() << "] Address:["
                << n.second << "]\n";
    }
  }

  uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
    auto res = lldiv(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
  }

  Workload get_chiplet_workload(
      Workload total_workload,
      uint64_t Yp,
      uint64_t Kp) {
    assert(Kp != 0);
    assert(Yp != 0);
    return Workload{ceil_div(total_workload.C, Kp),
                    ceil_div(total_workload.H, Yp),
                    total_workload.W};
  }

  Workload get_chiplet_sub_workload(
      Workload chiplet_workload,
      uint64_t Y2,
      uint64_t X2,
      uint64_t K2) {
    assert(K2 != 0);
    assert(Y2 != 0);
    assert(X2 != 0);
    return Workload{ceil_div(chiplet_workload.C, K2),
                    ceil_div(chiplet_workload.H, Y2),
                    ceil_div(chiplet_workload.W, X2)};
  }

  Point get_chiplet_out(
      Workload chiplet_sub_workload,
      uint64_t y2,
      uint64_t x2,
      uint64_t k2) {
    return Point{chiplet_sub_workload.C * k2,
                 chiplet_sub_workload.H * y2,
                 chiplet_sub_workload.W * x2};
  }

  Point chiplet_out_to_total_out(
      Workload chiplet_workload,
      uint64_t kp,
      uint64_t yp,
      Point point_out) {
    return Point{chiplet_workload.C * kp + point_out.C,
                 chiplet_workload.H * yp + point_out.Y,
                 point_out.X};
  }

  Point out_to_in(Point point_out, uint64_t stride_x, uint64_t stride_y) {
    return Point{0, point_out.Y * stride_y, point_out.X * stride_x};
  }
  
  void show_conv_param(torch::jit::Node* node) {
    num[0]++;
    layer_num++;
    layer_type = "conv" + to_string(num[0]);

    string layer_bf = "";
    if (layer_num-1) {
      layer_bf = "    form layer_num:" + to_string(layer_num_bf) + " type:" + layer_type_bf;
    }

    auto total_workload_out_shape = shape(node->output());
    auto total_workload_out = Workload{total_workload_out_shape[1],
                                       total_workload_out_shape[2],
                                       total_workload_out_shape[3]};

    param_conv.feature_map_size_x = total_workload_out.H;
    param_conv.feature_map_size_y = total_workload_out.W;

    const std::string& convname = layer_type;
    if (!BasicBlock_flag)
      const std::string& convname = node->inputs()[0]->node()->s(attr::name);

    std::cout << "layer_num:" << layer_num << " layer type:" << "conv" << num[0] << layer_bf << "\n";
    std::cout << convname << " param:\nin_channels:" << param_conv.in_channels
              << " out_channels:" << param_conv.out_channels << " kernel_size_x:"
              << param_conv.kernel_size_x << " kernel_size_y:" << param_conv.kernel_size_y
              << " stride_x:" << param_conv.stride_x << " stride_y:" << param_conv.stride_y
              << " dilation_x:"<< param_conv.dilation_x << " dilation_y:"
              << param_conv.dilation_y << " transposed:" << param_conv.transposed 
              << " feature_map_size_x:" << param_conv.feature_map_size_x 
              << " feature_map_size_y:" << param_conv.feature_map_size_y << std::endl;

    auto total_workload_in_shape = shape(node->inputs()[1]);
    auto total_workload_in = Workload{total_workload_in_shape[1],
                                      total_workload_in_shape[2],
                                      total_workload_in_shape[3]};
  
      auto knifeResult = NNKnife();
  
      auto chiplet_workload_out = get_chiplet_workload(
           total_workload_out, knifeResult.Yp, knifeResult.Kp);
      auto chiplet_sub_workload_out = get_chiplet_sub_workload(
           chiplet_workload_out,
           knifeResult.Y2,
           knifeResult.X2,
           knifeResult.K2);
      auto weight_address = allocateConv2dWeight(GetAttrValue, param_conv);
  
      for (uint64_t kp = 0; kp < knifeResult.Kp; kp++) {
        for (uint64_t yp = 0; yp < knifeResult.Yp; yp++) {
          uint64_t Chiplet_num = kp * knifeResult.Kp + yp;
          for (uint64_t y2 = 0; y2 < knifeResult.Y2; y2++) {
            for (uint64_t x2 = 0; x2 < knifeResult.X2; x2++) {
              for (uint64_t k2 = 0; k2 < knifeResult.K2; k2++) {
                auto chiplet_out =
                     get_chiplet_out(chiplet_sub_workload_out, y2, x2, k2);
                auto total_out = chiplet_out_to_total_out(
                     chiplet_workload_out, kp, yp, chiplet_out);
                auto total_in =
                     out_to_in(total_out, param_conv.stride_x, param_conv.stride_y);
                uint64_t act_addr;
                if ("input" == node->inputs()[1]->debugName()) {
                  act_addr = input_to_address(
                      total_workload_in, total_in.C, total_in.Y, total_in.X, address[node->inputs()[1]]);
                } else {
                  act_addr = activition_to_address(
                      total_workload_in,
                      knifeResult.Kp,
                      total_in.C,
                      total_in.Y,
                      total_in.X, address[node->inputs()[1]]);
                }
              }
            }
          }
        }
      }

    if (param_bn.flag) {
      std::cout << "bn param dimension:" << param_bn.dimen << std::endl;
      bzero(&param_bn, sizeof(BatchNormParameter));
    }
    if (param_relu.en) {
      std::cout << "relu param en:" << param_relu.en << " mode:" <<
      param_relu.mode << " param:" << param_relu.param << std::endl;
      param_relu.en = 0;
    }
    bzero(&param_conv, sizeof(Conv2dParameter));
    layer_num_bf  = layer_num;
    layer_type_bf = layer_type;
  }

  void node_backend(torch::jit::Node*& node, int cats_num) {
    auto kind = node->kind();
    if (kind == prim::GetAttr) {
      return;
    } else if (kind == aten::Int || kind == aten::view || kind == aten::flatten || \
               kind == aten::size || kind == aten::zeros || kind == aten::add_) {
      return;
    } else if (kind == prim::Constant || kind == prim::NumToTensor || \
               kind == prim::ListConstruct) {
      return;
    }

    else if (kind == aten::cat) {
      padding_num++;
      if (cats_num == padding_num) {
        auto param = parseCat(node);
        std::cout << "padding param: " << "param.dim_x:"  << param.dim_x 
                  << " param.dim_y:"  << param.dim_y 
                  << " param.dim_z:"  << param.dim_z
                  << " param.dim_w:"  << param.dim_w << std::endl;
        return;
      }
    }

    //relu layer in forward  && in BasicBlock && not in classifier
    else if (kind == aten::relu || 
      (is_module(node, "__torch__.torch.nn.modules.activation.ReLU"))) {

      //avoid relu layer in forward1
      auto nodeinput = node->inputs()[0]->node();
      if (relu_node_bk == nodeinput) {
          return;
      }

      param_relu.en = 1;
      param_relu.mode = "00";
      param_relu.param = "0_32";

      relu_node_bk = nodeinput;
      return;
    }

    else if (kind == aten::leaky_relu) {
      param_relu.en = 1;
      param_relu.mode = "10";
      param_relu.param = "0.01_32";

      return;
    }

    else if (kind == aten::tanh) {
      param_relu.en = 1;
      param_relu.mode = "11";
      param_relu.param = "0_32";

      return;
    }

    else if (kind == prim::CallMethod) {
      GetAttrValue = node->inputs()[0];
      if (is_module(node, "__torch__.torch.nn.modules.conv.Conv2d") ||
        is_module(node, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
        if (!num[0]) {
          if (param_conv.in_channels) {
              show_conv_param(node_back);
          }
          layer_type_bf = "conv1";
        } else {
          if (is_module(node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
              is_module(node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
              show_conv_param(node_back);
          }
        }
        param_conv = parseConv2d(node);
        node_back  = node;
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.batchnorm.BatchNorm2d")) {
        string::size_type idx1, idx2;
        idx1 = layer_type_bf.find("conv");
        idx2 = layer_type_bf.find("pool");
        if(idx1 != string::npos or idx2 != string::npos) { 
            param_bn = parseBatchNorm(node);
        } else {
          std::cout << "before BatchNorm layer is wrong..." << std::endl;
        }
        if(downsample_flag > 0) {
           downsample_flag--;
        }
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d") ||
               is_module(node, "__torch__.torch.nn.modules.pooling.AvgPool2d")) {
        if (is_module(node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
            show_conv_param(node_back);
        }
        std::string pad = "avg";
        if (is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d")) {
          pad = "max",
          BasicBlock_cnt++;
        }

        num[1]++;
        layer_num++;
	node_back  = node;
        string layer_bf = "";
        layer_type = pad + "pool_" + to_string(num[1]);
        if (layer_num-1) {
          layer_bf = "    form layer_num:" + to_string(layer_num_bf) + " type:" + layer_type_bf;
        }
        auto param = parsePool2d(node);
        auto size = param.kernel_size_x * param.kernel_size_y;
        std::string poolname = layer_type;
        if (!BasicBlock_flag) {
            poolname = node->inputs()[0]->node()->s(attr::name);
            layer_type = poolname;
        }
        if (BasicBlock_total == to_string(BasicBlock_cnt-1)) {
          BasicBlock_flag = false;
        }
        std::cout << "layer_num:" << layer_num << " layer type:" << layer_type  << layer_bf << "\n";
        std::cout << poolname << " param: pool_size:" << size - 1 << " kernel_size_x:" << param.kernel_size_x
                  << " kernel_size_y:" << param.kernel_size_y << " Pooling_en:1" << " oprands:" << 1.0 / size 
                  << " stride_x:" << param.stride_x << " stride_y:" << param.stride_y << std::endl;
        layer_num_bf  = layer_num;
        layer_type_bf = layer_type;
        return;
      }

      else if (is_module(node, "__torch__.BasicBlock") ||
               is_module(node, "__torch__.Bottleneck") ||
               is_module(node, "__torch__.torchvision.models.resnet.BasicBlock") ||
               is_module(node, "__torch__.torchvision.models.resnet.Bottleneck")) {
        BasicBlock_cnt++;
        BasicBlock_flag = true;
        BasicBlock_node(node->outputs()[0]);
        return;
      }

      else if (is_module(node, "__torch__.quant_layer.QuantLayer")) {
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.dropout.Dropout")) {
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.linear.Identity")) {
        if (is_module(node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
            show_conv_param(node_back);
        }
        num[3]++;
        layer_num++;
        node_back  = node;
        string layer_bf = "";
        layer_type = "Maxpool" + to_string(num[3]);
        if (layer_num-1) {
            layer_bf = "    form layer_num:" + to_string(layer_num_bf) + " type:" + layer_type_bf;
        }
        std::cout << "layer_num:" << layer_num << " layer type:" << layer_type  << layer_bf << "\n";
        const std::string& poolname = layer_type;
        std::cout << poolname << "Identity layer" << std::endl;
        layer_num_bf  = layer_num;
        layer_type_bf = layer_type;
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.container.Sequential")) {
        Sequential_node(node->outputs()[0]);
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d")) {
        if (is_module(node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
            show_conv_param(node_back);
        }

        num[3]++;
        layer_num++;
        node_back  = node;
        string layer_bf = "";
        layer_type = "AdaptAvgPool" + to_string(num[3]);
        if (layer_num-1) {
          layer_bf = "    form layer_num:" + to_string(layer_num_bf) + " type:" + layer_type_bf;
        }
        auto param = parseAdapt(node);
        const std::string& poolname = layer_type;
        std::cout << "layer_num:" << layer_num << " layer type:" << layer_type  << layer_bf << "\n";
        std::cout << poolname << " param: output_size_x " << param.output_size_x << " output_size_y:"
                  << param.output_size_y << std::endl;
        layer_num_bf  = layer_num;
        layer_type_bf = layer_type;
        return;
      }

      //relu layer in classifier
      else if (is_module(node, "__torch__.torch.nn.modules.activation.ReLU") and \
        !BasicBlock_flag and num[2]) {
        std::cout << "fc layer has ReLU" << std::endl;
        return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.linear.Linear")) {
        if (is_module(node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
            show_conv_param(node_back);
        }
        BasicBlock_flag = false;
        num[2]++;
        layer_num++;
	node_back  = node;
        layer_type = "fc" + to_string(num[2]);
        string layer_bf = "";
        if (layer_num-1) {
            layer_bf = "    form layer_num:" + to_string(layer_num_bf) + " type:" + layer_type_bf;
        }
        auto param = parseLinear(node);
        std::cout << "layer_num:" << layer_num << " layer type:" << layer_type  << layer_bf << "\n";
        std::cout << layer_type << " param:" << "in_features_x:" << param.in_features_x 
                  << " in_features_y:" << param.in_features_y << " out_features_x:" 
                  << param.out_features_x << " out_features_y:" << param.out_features_y << std::endl;
        layer_num_bf  = layer_num;
        layer_type_bf = layer_type;
        return;
      }

      auto type = GetAttrValue->type()->cast<c10::ClassType>();
      TORCH_CHECK(type && type->name());
      std::cout << "compiler.cpp error:" << type->name()->qualifiedName() << std::endl;

      TORCH_CHECK(false);
      return;
    }
  }

  void backend() {
    auto cats_num = 0;
    auto nodes = module.get_method("forward").graph()->nodes();

    for (auto&& node : nodes) {
      if (node->kind() == aten::cat) {
        cats_num++;
      }
    }
    for (auto&& node : nodes) {
      node_backend(node, cats_num);
    }
    if (param_conv.in_channels) {
      show_conv_param(node_back);
    }
  }
};
} // namespace fun
