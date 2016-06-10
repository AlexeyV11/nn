classdef LossTriplet < LossInterface
    properties
    end
    
    properties (Access = 'private')        
        margin
    end
    
    methods (Access = 'public')
        function obj = LossTriplet(margin)
            obj.margin = margin;
        end
        
        function [result] = computeLoss(obj, anchor, positive, negative)
            %anchor = obj.mynorm(anchor);
            %positive = obj.mynorm(positive);
            %negative = obj.mynorm(negative);

            dists_ap = obj.mydist(anchor, positive);
            dists_an = obj.mydist(anchor, negative);

            result = max(obj.margin - dists_an + dists_ap,0);
        end
        
        function [anchor_derivative, positive_derivative, negative_derivative] = computeDerivative(obj, anchor, positive, negative)
            anchor_derivative = 2 * (negative - positive);
            positive_derivative = 2 * (positive - anchor);
            negative_derivative = 2 * (anchor - negative);
            
            loss = computeLoss(obj, anchor, positive, negative);
            
            anchor_derivative(loss == 0, :) = 0;
            positive_derivative(loss == 0, :) = 0;
            negative_derivative(loss == 0, :) = 0;
        end
        
        function [dists] = mydist(obj, anchor_feats, positive_feats)
            diff = (anchor_feats - positive_feats);
            dists = sqrt(sum(diff.^2,2));
        end
        
        %function [featsNormalized] = mynorm(obj, feats)
        %    factor = sqrt(sum(feats.*feats,2));
        %    featsNormalized = feats ./ repmat(factor,1,size(feats,2));
        %end

    end    
end


