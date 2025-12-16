# include/adaptive_quant.py
class DynamicQuantizer:
    def quantize_layer(self, layer, target_bits, importance_score):
        """Dynamic precision based on sensitivity"""
        if importance_score > 0.9:
            return layer  # Keep FP16 for critical layers
        elif importance_score > 0.7:
            return quantize_to_int8(layer)
        else:
            return quantize_to_int4(layer)
    
    def calculate_importance(self, layer, gradients):
        """How sensitive is this layer to quantization?"""
        # Fisher information-based importance
        fisher_info = torch.mean(gradients ** 2)
        return fisher_info / torch.max(fisher_info)
