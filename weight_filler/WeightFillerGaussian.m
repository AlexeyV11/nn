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
            weights = random('normal',0,obj.variance,size);
        end
        
        
    end    
end

