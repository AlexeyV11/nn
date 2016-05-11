classdef GradientUpdaterInterface < handle
    properties
    end
    
    properties (Access = 'private')        
    end
    
    methods(Abstract, Access = public)
        [] = setLearningRate(obj, lr)
        [weights] = update(obj, weights, dw)
    end    
end

