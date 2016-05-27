function [ output_args ] = trainTripletLossNetwork( nn, epochs, minibatchSize, margin, dataProvider)
    
    
    lossTriplet = LossTriplet(margin);
    
    for epoch = 1:epochs
        itersCount = floor(dataProvider.getSamplesCount()/minibatchSize);
        
        lossEpoch = [];
        
        for iters = 1:itersCount 
        
            feats = struct();
            [feats.anchor, feats.positive, feats.negative] = dataProvider.getMinibatch(minibatchSize);
    
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
            
            disp(sum(loss) / numel(loss));        
        end

        disp(['epoch : ' num2str(epoch) ' loss : ' num2str(sum(lossEpoch) / numel(lossEpoch))]);
    end
end

