classdef LayerConv2 < LayerInterface
    properties
    end
    
    properties (Access = 'public')        
        weights;
        depth;
        kernelSize;
        inputDimensions;
        outputDimensions;
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
            obj.activationsPrev = activationsPrev;
            
            obj.outputDimensions = [obj.inputDimensions(1),obj.inputDimensions(2),obj.depth];
            convs = zeros(obj.outputDimensions);
            
            numSamples = size(activationsPrev,1);
            result = zeros(numSamples, numel(convs));
            for sampleIndex=1:numSamples
                inputFeature = reshape(activationsPrev(sampleIndex,:), obj.inputDimensions);
                for i=1:obj.depth
                    convs(:,:,i) = sum(convn(inputFeature, obj.weights{i},'same'),3);
                end
                
                result(sampleIndex,:) = reshape(convs,[1 numel(convs)]);
            end
        end
        
        function [gradientToPrev] = backPropagate(obj, gradientToCurrent, gradientUpdater)
            % http://www.slideshare.net/kuwajima/cnnbp
            
            numSamples = size(obj.activationsPrev,1);
            gradientToPrev = zeros(size(gradientToCurrent));
            for sampleIndex=1:numSamples
        
                % update current weights
        
                inputFeature = reshape(obj.activationsPrev(sampleIndex,:), obj.inputDimensions);
                gradient = reshape(gradientToCurrent(sampleIndex,:), obj.outputDimensions);
        
                dweights = cell(1,obj.outputDimensions(3));
                for i=1:obj.outputDimensions(3)
                    %weights{end+1} = convn(inputFeature,gradient(:,:,i),'same');
                    pad = (obj.kernelSize - 1) / 2;
                    conv_res = convn(padarray(inputFeature,[pad pad],0,'both'),gradient(:,:,i),'val');
                    dweights{i} = sum(conv_res,3);
                end
                
                % backpropogate gradient
                % todo
                
                
                %%%
                for i=1:obj.outputDimensions(3)
                    obj.weights{i} = gradientUpdater.update(obj.weights{i}, flip(flip(dweights{i},1),2));
                end
                
                
            end
            
            
        end
        
    end    
end

