#include "core/rcc.hpp"

#include <unordered_set>

#include "core/coreset.hpp"

namespace kmeans::core {

namespace {
std::unique_ptr<RCCNode> mergeNodes(std::unique_ptr<RCCNode> nodeA, std::unique_ptr<RCCNode> nodeB) {
    if (!nodeA) {
        return nodeB;
    }

    if (!nodeB) {
        return nodeA;
    }

    Coreset merged = mergeCoresets(nodeA->coreset, nodeB->coreset);
    auto parent = std::make_unique<RCCNode>(merged);
    parent->left = std::move(nodeA);
    parent->right = std::move(nodeB);

    return parent;
}
} // namespace

RCC::RCC(int maxLevels) : max_levels(maxLevels) {
    levels.resize(std::max(1, max_levels));
}

void RCC::insertLeaf(const Coreset& leafCoreset) {
    auto carry = std::make_unique<RCCNode>(leafCoreset);

    if (levels.empty()) {
        levels.resize(std::max(1, this->max_levels));
    }

    for (auto& level : levels) {
        if (!level) {
            level = std::move(carry);
            return;
        }
        carry = mergeNodes(std::move(level), std::move(carry));
    }

    levels.back() = std::move(carry);
}



Coreset RCC::getRootCoreset() const {
    Coreset final_coreset;
    bool empty = true;
    for (const auto& node : levels) {
        if (node) {
            if (empty) {
                final_coreset = node->coreset;
                empty = false;
            } else {
                final_coreset = mergeCoresets(final_coreset, node->coreset);
            }
        }
    }
    return final_coreset;
}

void RCC::clear() {
    levels.clear();
    levels.resize(std::max(1, max_levels));
}

} // namespace kmeans::core