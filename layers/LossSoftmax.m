classdef LossSoftmax < LossInterface
    properties
    end
    
    properties (Access = 'private')        
        featureDimension;
    end
    
    methods (Access = 'public')
        function obj = LossSoftmax(dimension)
            obj.featureDimension = dimension;
        end
        
        function [result] = feedForward(obj, activationsPrev, activationsTarget)
            
            % http://cs231n.github.io/linear-classify/#softmax
            score_new = activationsPrev - repmat(max(activationsPrev')', 1, size(activationsPrev,2));
            probabilities = exp(score_new)./repmat(sum(exp(score_new'))', 1, size(activationsPrev,2));
            
            result = -log(probabilities(boolean(activationsTarget)));
        end
        
        function [gradientToPrev] = backPropagate(obj, activationsPrev, activationsTarget)
            gradientToPrev = (activationsPrev - activationsTarget);
        end
        
    end    
end

