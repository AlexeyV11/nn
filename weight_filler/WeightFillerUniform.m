classdef WeightFillerUniform < WeightFillerInterface
    properties
    end
    
    properties (Access = 'private')
        epsilon
    end
    
    methods (Access = 'public')
        function obj = WeightFillerUniform(epsilon)
            obj.epsilon = epsilon;
        end
        
        function [weights] = generateWeights(obj, size)
            weights = rand(size) * (2*obj.epsilon) - obj.epsilon;
        end
        
        
    end    
end

