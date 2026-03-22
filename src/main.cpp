#include <iostream>
#include <vector>
#include <string>

#include "core/coreset.hpp"
#include "common/config.hpp"
#include "io/application.hpp"

using namespace kmeans;

// Entry point
int main() 
{
    std::cout << "Starting K-Means Clustering for Thesis - ImGui Application..." << std::endl;
    
    io::Application app;
    app.run();
    
    return 0;
}