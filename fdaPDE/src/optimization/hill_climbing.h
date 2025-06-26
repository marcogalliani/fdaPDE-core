// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __HILL_CLIMBING_H__
#define __HILL_CLIMBING_H__

#include "header_check.h"

namespace fdapde {

// searches for the point in a given grid minimizing a given nonlinear objective
// N: dimensions of the grid (number of inputs of the objective function)
template <int N, typename... GridDimensionsT>
    requires(sizeof...(GridDimensionsT) == N)
    class HillClimbing
    {
   private:
    //the grid is a collection of 1D arrays
    using grid_t = std::tuple<std::vector<GridDimensionsT>...>; //cannot be std::array<std::vector<double>, N>> since types could be different
    grid_t grid_;
    using grid_index_t = std::array<int, N>;
    using grid_size_t = std::array<int, N>;
    grid_size_t grid_size_;

    //method to extract the element of the grid from an index
    template<std::size_t... Is>
    constexpr std::tuple<GridDimensionsT...> get_grid_point_impl(grid_index_t point_index,  std::index_sequence<Is...>){
        return std::make_tuple(std::get<Is>(grid_)[point_index[Is]]...);
    }
    constexpr std::tuple<GridDimensionsT...> get_grid_point(grid_index_t point_index) {
        return get_grid_point_impl(point_index,std::make_index_sequence<N>{});
    }
    //store the optimum
    std::array<int,N> opt_index_;
    double optimal_value_;   // objective value at optimum
    //store the explored values in unordered map to gurantee fast access
    //-> we need a custom hasher to convert std::array<int,N> to int(keys)
    struct ArrayHasher {
        size_t operator()(const grid_size_t& key) const {
            size_t hash = 0;
            for (auto& k : key) hash ^= std::hash<int>{}(k) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            return hash;
        }
    };
    //-> define the unordered map storing explored values
    std::unordered_map<grid_index_t, double, ArrayHasher> explored_values_; //should be something like: key->grid_index_t, value->double
    //parameters
    int max_iter_ = 100;
   public:
    HillClimbing(std::vector<GridDimensionsT>... grids) : grid_(grids...), grid_size_{grids.size()...} {}

    template <typename ObjectiveT>
    std::tuple<GridDimensionsT...> optimize(ObjectiveT&& objective) {
        //iterate until the optimum is found
        int n_iter = 0;
        grid_index_t curr_index;
        curr_index.fill(0);
        opt_index_.fill(0);
        optimal_value_ = std::apply(objective, get_grid_point(curr_index));
        explored_values_[curr_index] = optimal_value_;

        bool opt_found = false;
        while(!opt_found && n_iter < max_iter_) {
            bool improvement_found = false;
            for (int i = 0; i < N && !improvement_found; ++i) {
                for (int d : {-1, 1}) {
                    grid_index_t neighbor = curr_index;
                    neighbor[i] += d;
                    // validate bounds
                    if (neighbor[i] < 0 || neighbor[i] >= grid_size_[i]) continue;
                    // compute objective value at the next index (rely on the already computed ones if present)
                    double value;
                    if (auto it = explored_values_.find(neighbor); it != explored_values_.end()) {
                        value = it->second;
                    } else {
                        value = std::apply(objective, get_grid_point(neighbor));
                        explored_values_[neighbor] = value;
                    }
                    // check if the objective value at the neighbor index is better
                    if (value < optimal_value_) {
                        opt_index_ = neighbor;
                        optimal_value_ = value;
                        curr_index = neighbor;
                        improvement_found = true;
                        break; // exit early if it is
                    }
                }
            }
            opt_found = !improvement_found; //if exit the loop without having found improvements I'm in the optimum
            n_iter++;
        }
        return get_grid_point(opt_index_);
    }
    //observers

};

}   // namespace fdapde

#endif   // __HILL_CLIMBING_H__
