function [ output_args ] = trainTripletLossNetwork( nn, epochs, minibatchSize, margin, dataProvider)
    lossTriplet = LossTriplet(margin);
    
    for epoch = 1:epochs
        itersCount = floor(dataProvider.getSamplesCount()/minibatchSize);
        lossEpoch = 0;
    
        for iters = 1:itersCount 
            feats = getProperTriplets( nn, minibatchSize, margin, dataProvider);
            
            %feats.anchor =      bsxfun(@rdivide, feats.anchor,      sqrt(sum(abs(feats.anchor).^2,2)));
            %feats.positive =    bsxfun(@rdivide, feats.positive,    sqrt(sum(abs(feats.positive).^2,2)));
            %feats.negative =    bsxfun(@rdivide, feats.negative,    sqrt(sum(abs(feats.negative).^2,2)));
            
            anchorOutputs = nn.forwardPropogate(feats.anchor);
            positiveOutputs = nn.forwardPropogate(feats.positive);
            negativeOutputs = nn.forwardPropogate(feats.negative);
            
            loss = lossTriplet.computeLoss(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
            [anchor_derivative, positive_derivative, negative_derivative] = lossTriplet.computeDerivative(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
        
            anchorGrads = nn.backPropagate1(anchorOutputs, anchor_derivative);
            positiveGrads = nn.backPropagate1(positiveOutputs, positive_derivative);
            negativeGrads = nn.backPropagate1(negativeOutputs, negative_derivative);
            
            grad = cell(1,numel(anchorGrads));
            
            for i=1:numel(grad)
                if(~isempty(anchorGrads{i}))
                    grad{i} = anchorGrads{i} + positiveGrads{i} + negativeGrads{i};
                end
            end
            
            nn.updateWeights(grad);
            
            %disp(['epoch : ' num2str( sum(loss) / numel(loss))]);
            lossEpoch = lossEpoch + sum(loss) / numel(loss);
        end

        disp(['epoch : ' num2str(epoch) ' loss : ' num2str(lossEpoch / itersCount)]);
    end
end


function [ feats ] = getProperTriplets( nn, minibatchSize, margin, dataProvider)
    feats = struct();
    [feats.anchor, feats.positive, feats.negative] = dataProvider.getMinibatch(minibatchSize);
end


