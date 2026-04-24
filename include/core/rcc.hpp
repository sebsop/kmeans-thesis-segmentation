#pragma once

#include <memory>
#include <vector>

#include "core/coreset.hpp"

namespace kmeans::core {

// A node in the RCC tree, holding a coreset and unique_ptrs to left and right children
// If leaf, both children are nullptr
struct RCCNode {
    Coreset coreset;
    std::unique_ptr<RCCNode> left;
    std::unique_ptr<RCCNode> right;

    explicit RCCNode(const Coreset& c) : coreset(c) {}
};

// The RCC (Recursive Cached Coreset Tree) structure for managing and merging coresets
class RCC {
  private:
    std::vector<std::unique_ptr<RCCNode>> levels;
    int max_levels = 8;

  public:
    // Default constructor and destructor
    RCC() = default;
    ~RCC() = default;

    // Initialize RCC with a maximum number of levels
    // Args:
    //   maxLevels: maximum number of levels in the RCC tree
    //			  (higher means larger memory usage but potentially better accuracy)
    explicit RCC(int maxLevels);

    // Insert a new leaf coreset into the RCC tree, merging as necessary to maintain structure
    //
    // Args:
    //   leafCoreset: the new coreset to insert as a leaf
    void insertLeaf(const Coreset& leafCoreset);

    // Get the coreset at the root of the RCC tree
    //
    // Returns:
    //   The root Coreset, or an empty Coreset if the tree is empty.
    [[nodiscard]] Coreset getRootCoreset() const;

    // Clear memory and reset the tree state
    void clear();
};

} // namespace kmeans::core