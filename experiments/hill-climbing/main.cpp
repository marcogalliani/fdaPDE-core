//
// Created by Marco Galliani on 26/06/25.
//
#include <fdaPDE/core.h>
using namespace fdapde;

#include "nlohmann/json.hpp"
using nlohmann::json;

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::vector<double> x_vals = {0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> y_vals = {1.0, 1.5, 2.0, 2.5, 3.0};
    std::vector<int> z_vals = {0, 5, 10, 15, 20};

    HillClimbing<3, double, double, int> hc(x_vals, y_vals, z_vals);

    // Define a plateau objective function
    auto objective = [](double x, double y, int z) {
        if (std::abs(x - 1.0) <= 0.5 && std::abs(y - 2.0) <= 0.5 && std::abs(z - 10) <= 5)
            return 0.0;
        return std::pow(x - 1.0, 2) + std::pow(y - 2.0, 2) + std::abs(z - 10);
    };

    auto result = hc.optimize(objective);

    double x_opt = std::get<0>(result);
    double y_opt = std::get<1>(result);
    int z_opt = std::get<2>(result);

    std::cout << "\nFound optimal point in plateau:\n";
    std::cout << "x = " << x_opt << "\n";
    std::cout << "y = " << y_opt << "\n";
    std::cout << "z = " << z_opt << "\n";

    // The result can be anywhere on the plateau
    assert(std::abs(x_opt - 1.0) <= 0.5);
    assert(std::abs(y_opt - 2.0) <= 0.5);
    assert(std::abs(z_opt - 10) <= 5);

    std::cout << "✅ Test passed: Found point inside the flat optimal region.\n";
    return 0;
}