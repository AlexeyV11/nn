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
            % https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
            activationsNorm = activationsPrev./repmat(sum(activationsPrev,2),[1,size(activationsPrev,2)]);
            activationsLog = -log(activationsNorm);
            result = activationsLog(boolean(activationsTarget));
            result = min(result, 20.0);
        end
        
        function [gradientToPrev] = backPropagate(obj, activationsPrev, activationsTarget)
            gradientToPrev = (activationsPrev - activationsTarget);
        end
        
    end    
end

