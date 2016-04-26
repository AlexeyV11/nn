classdef WeightFillerInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods(Abstract, Access = public)
        [weights] = generateWeights(obj, size)
    end    
end

