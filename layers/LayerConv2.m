classdef LayerConv2 < LayerInterface
    properties
    end
    
    properties (Access = 'private')        
        weights;
        depth;
        kernelSize;
        inputDimensions;
        activationsPrev;
    end
    
    methods (Access = 'public')
        function obj = LayerConv2(inputDimensions, kernelSize, depth, weightFiller)
            obj.kernelSize = kernelSize;
            obj.depth = depth;
            obj.inputDimensions = inputDimensions;
            
            for i=1:depth
                obj.weights{end+1} = weightFiller.generateWeights([kernelSize, kernelSize]);
            end
        end
        
        function [result] = feedForward(obj, activationsPrev)
            convs = zeros(obj.inputDimensions(1),obj.inputDimensions(2),obj.depth);
            
            numSamples = size(activationsPrev,1);
            result = zeros(numSamples, numel(convs));
            for sampleIndex=1:numSamples
                img = reshape(activationsPrev(sampleIndex,:), obj.inputDimensions);
                for i=1:obj.depth
                    convs(:,:,i) = sum(convn(img, obj.weights{i},'same'),3);
                end
                
                result(sampleIndex,:) = reshape(convs,[1 numel(convs)]);
            end
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, gradientUpdater)
            % backpropogate gradient
            
            % update current weights
        end
        
    end    
end

