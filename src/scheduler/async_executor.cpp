// scr/scheduler/async_executor.cpp
class AsyncExecutor {
    std::vector<cudaStream_t> compute_streams;
    cudaStream_t transfer_stream;
    std::queue<AsyncTask> task_queue;
    
    void execute_with_overlap(const ModelLayer& layer, const Tensor& input) {
        // 1. Start async transfer of next layer if not resident
        if (!next_layer_resident) {
            cudaMemcpyAsync(..., transfer_stream);
        }
        
        // 2. Compute current layer on compute stream
        cudaLaunchKernel(layer.kernel, compute_streams[0], ...);
        
        // 3. On completion, signal transfer stream can proceed
        cudaEventRecord(compute_done, compute_streams[0]);
        cudaStreamWaitEvent(transfer_stream, compute_done);
        
        // 4. While transferring, CPU can prepare next operation
        if (layer_needs_cpu_preprocess) {
            std::thread cpu_worker([&]() {
                prepare_next_operation();
            });
        }
    }
};
