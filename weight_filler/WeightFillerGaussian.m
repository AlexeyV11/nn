classdef WeightFillerGaussian < WeightFillerInterface
    properties
    end
    
    properties (Access = 'private')
        variance
    end
    
    methods (Access = 'public')
        function obj = WeightFillerGaussian(variance)
            obj.variance = variance;
        end
        
        function [weights] = generateWeights(obj, size)
            weights = sqrt(obj.variance)*randn(size);
        end
        
        
    end    
end

