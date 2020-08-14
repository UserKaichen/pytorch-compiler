#include <assert.h>
#include <iostream>

using namespace std;

const uint64_t exp_2_31 = (uint64_t)1 << 31;
const uint64_t activition_bus_width = 16;
const uint64_t input_bus_width = 8;
const uint64_t weight_bus_width = 8;

struct Workload {
  uint64_t C;
  uint64_t H;
  uint64_t W;
};

uint64_t align(uint64_t x, int align_factor) {
  auto div_result = div(x, align_factor);
  if (div_result.rem)
    return (div_result.quot + 1) * align_factor;
  else
    return x;
}

struct Point {
  uint64_t C;
  uint64_t Y;
  uint64_t X;
};

uint64_t activition_to_address(
    Workload workload,
    uint64_t Kp,
    uint64_t C,
    uint64_t H,
    uint64_t W) {
  uint64_t address = 0;
  int align_factor;

  if (Kp == 1)
    align_factor = 32;
  else
    align_factor = 64;

  workload.C = align(workload.C, align_factor);

  uint64_t half_total_channel = workload.C / 2;

  if (C >= half_total_channel) {
    address = exp_2_31;
    C -= half_total_channel;
  }

  return address + (C / activition_bus_width) * workload.H * workload.W +
      H * workload.W + W;
}

void test_activition_to_address() {
  auto workload = Workload{96, 512, 512};

  assert(activition_to_address(workload, 1, 0, 0, 0) == 0);
  assert(activition_to_address(workload, 1, 15, 0, 0) == 0);
  assert(activition_to_address(workload, 1, 0, 0, 1) == 1);
  assert(activition_to_address(workload, 1, 12, 0, 511) == 511);
  assert(activition_to_address(workload, 1, 0, 1, 0) == 512);
  assert(activition_to_address(workload, 1, 13, 511, 511) == 262143);
  assert(activition_to_address(workload, 1, 16, 0, 0) == 262144);
  assert(activition_to_address(workload, 1, 20, 511, 511) == 524287);
  assert(activition_to_address(workload, 1, 33, 511, 511) == 786431);
  assert(activition_to_address(workload, 1, 50, 0, 0) == 2147483648);
  assert(activition_to_address(workload, 1, 90, 511, 511) == 2148270079);

  assert(activition_to_address(workload, 4, 0, 0, 0) == 0);
  assert(activition_to_address(workload, 4, 15, 0, 0) == 0);
  assert(activition_to_address(workload, 4, 0, 0, 1) == 1);
  assert(activition_to_address(workload, 4, 12, 0, 511) == 511);
  assert(activition_to_address(workload, 4, 0, 1, 0) == 512);
  assert(activition_to_address(workload, 4, 13, 511, 511) == 262143);
  assert(activition_to_address(workload, 4, 16, 0, 0) == 262144);
  assert(activition_to_address(workload, 4, 20, 511, 511) == 524287);
  assert(activition_to_address(workload, 4, 50, 511, 511) == 1048575);
  assert(activition_to_address(workload, 4, 70, 0, 0) == 2147483648);
  assert(activition_to_address(workload, 4, 120, 511, 511) == 2148532223);
}

uint64_t input_to_address(
    Workload workload,
    uint64_t C,
    uint64_t H,
    uint64_t W) {
  uint64_t address = 0;
  const int align_factor = 8;

  workload.C = align(workload.C, align_factor);

  auto div_result = div(workload.H, 2);
  uint64_t half_total_height = div_result.quot + (div_result.rem ? 1 : 0);

  if (H >= half_total_height) {
    address = exp_2_31;
    H -= half_total_height;
  }

  return address + (C / activition_bus_width) * workload.H * workload.W +
      H * workload.W + W;
}

void test_input_to_address() {
  auto workload = Workload{3, 512, 512};

  assert(input_to_address(workload, 0, 0, 0) == 0);
  assert(input_to_address(workload, 7, 0, 0) == 0);
  assert(input_to_address(workload, 0, 0, 1) == 1);
  assert(input_to_address(workload, 7, 0, 511) == 511);
  assert(input_to_address(workload, 0, 1, 0) == 512);
  assert(input_to_address(workload, 7, 255, 511) == 131071);
  assert(input_to_address(workload, 6, 256, 0) == 2147483648);
  assert(input_to_address(workload, 0, 511, 511) == 2147614719);

  auto workload_2 = Workload{3, 511, 511};

  assert(input_to_address(workload_2, 0, 0, 0) == 0);
  assert(input_to_address(workload_2, 7, 0, 0) == 0);
  assert(input_to_address(workload_2, 0, 0, 1) == 1);
  assert(input_to_address(workload_2, 7, 0, 510) == 510);
  assert(input_to_address(workload_2, 7, 1, 0) == 511);
  assert(input_to_address(workload_2, 7, 255, 510) == 130815);
  assert(input_to_address(workload_2, 6, 256, 0) == 2147483648);
  assert(input_to_address(workload_2, 0, 510, 510) == 2147613952);
}

struct WeightWorkload {
  uint64_t Cout;
  uint64_t Cin;
  uint64_t H;
  uint64_t W;
};

uint64_t weight_to_address(
    WeightWorkload workload,
    uint64_t Cout,
    uint64_t Cin,
    uint64_t H,
    uint64_t W) {
  auto raw_Cout_coefficient =
      (workload.Cin / weight_bus_width) * workload.H * workload.W;
  auto Cout_coefficient = raw_Cout_coefficient + raw_Cout_coefficient % 2;

  return 2 * (Cout + 1) + Cout * Cout_coefficient +
      (Cin / weight_bus_width) * workload.H * workload.W + H * workload.W + W;
}

void test_weight_to_address() {
  auto workload = WeightWorkload{512, 512, 3, 3};

  assert(weight_to_address(workload, 0, 0, 0, 0) == 2);
  assert(weight_to_address(workload, 0, 7, 0, 1) == 3);
  assert(weight_to_address(workload, 0, 7, 0, 2) == 4);
  assert(weight_to_address(workload, 0, 7, 1, 0) == 5);
  assert(weight_to_address(workload, 0, 0, 2, 2) == 10);
  assert(weight_to_address(workload, 0, 8, 0, 0) == 11);
  assert(weight_to_address(workload, 0, 510, 2, 2) == 577);
  assert(weight_to_address(workload, 1, 7, 0, 0) == 580);
  assert(weight_to_address(workload, 1, 510, 2, 2) == 1155);
  assert(weight_to_address(workload, 2, 3, 0, 0) == 1158);
  assert(weight_to_address(workload, 511, 510, 2, 2) == 295935);

  auto workload_2 = WeightWorkload{512, 504, 3, 3};

  assert(weight_to_address(workload_2, 0, 0, 0, 0) == 2);
  assert(weight_to_address(workload_2, 0, 7, 0, 1) == 3);
  assert(weight_to_address(workload_2, 0, 7, 0, 2) == 4);
  assert(weight_to_address(workload_2, 0, 7, 1, 0) == 5);
  assert(weight_to_address(workload_2, 0, 0, 2, 2) == 10);
  assert(weight_to_address(workload_2, 0, 8, 0, 0) == 11);
  assert(weight_to_address(workload_2, 0, 501, 2, 2) == 568);
  assert(weight_to_address(workload_2, 1, 7, 0, 0) == 572);
  assert(weight_to_address(workload_2, 1, 500, 2, 2) == 1138);
  assert(weight_to_address(workload_2, 2, 3, 0, 0) == 1142);
  assert(weight_to_address(workload_2, 511, 500, 2, 2) == 291838);
}