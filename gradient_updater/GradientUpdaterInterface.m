classdef GradientUpdaterInterface
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods(Abstract, Access = public)
        [weights] = update(obj, weights, dw)
    end    
end

