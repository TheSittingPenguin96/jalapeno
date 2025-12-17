#include "jalapeno/tensor.hpp"
#include "jalapeno/memory_manager.hpp"
#include <iostream>

using namespace jalapeno;

int main() {
    std::cout << "Jalapeño Test Program\n";
    
    // Initialize memory manager
    auto& mm = MemoryHierarchyManager::instance();
    
    std::map<MemoryTier, size_t> capacities = {
        {MemoryTier::TIER_0_GPU_HBM, 8UL * 1024 * 1024 * 1024},  // 8GB
        {MemoryTier::TIER_2_CPU_RAM, 16UL * 1024 * 1024 * 1024}, // 16GB
    };
    
    mm.initialize(capacities);
    
    // Create a tensor
    auto tensor = mm.create_tensor({1024, 1024}, DataType::FP32, 
                                   MemoryTier::TIER_2_CPU_RAM);
    
    std::cout << "Tensor created: " 
              << tensor->metadata().bytes / (1024*1024) << " MB\n";
    
    // Migrate to GPU
    std::cout << "Migrating tensor to GPU...\n";
    mm.migrate_tensor(tensor, MemoryTier::TIER_0_GPU_HBM);
    
    // Get stats
    auto stats = mm.get_stats();
    std::cout << "GPU used: " << stats.gpu_used / (1024*1024) << " MB\n";
    std::cout << "GPU total: " << stats.gpu_total / (1024*1024) << " MB\n";
    
    // Simulate access pattern
    for (int i = 0; i < 10; i++) {
        tensor->record_access();
        mm.register_access_pattern(tensor, tensor);  // Self-reference for demo
    }
    
    std::cout << "Tensor temperature: " << tensor->access_temperature() << "\n";
    
    // Test prefetch prediction
    auto predicted = mm.predict_next_accesses(tensor);
    std::cout << "Predicted next accesses: " << predicted.size() << " tensors\n";
    
    std::cout << "Test passed!\n";
    return 0;
}
