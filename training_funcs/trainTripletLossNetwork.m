function [ output_args ] = trainTripletLossNetwork( nn, epochs, minibatchSize, margin, dataProvider)
    lossTriplet = LossTriplet(margin);

    
    disp(['epoch : ' num2str(0) ' loss : ' num2str(compute_epoch_loss(nn, minibatchSize * 128, margin, dataProvider))]);

    for epoch = 1:epochs
        itersCount = floor(dataProvider.getSamplesCount()/minibatchSize);
        lossEpoch = 0;
    
        for iters = 1:itersCount 
            feats = getProperTriplets( nn, minibatchSize, margin, dataProvider, lossTriplet);
            
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
        end

        
        disp(['epoch : ' num2str(epoch) ' loss : ' num2str(compute_epoch_loss(nn, minibatchSize * 128, margin, dataProvider))]);
    end
end

function [loss] = compute_epoch_loss(nn, tripletCount, margin, dataProvider)
    lossTriplet = LossTriplet(margin);

    feats = struct();
    [feats.anchor, feats.positive, feats.negative] = dataProvider.getMinibatch(tripletCount);

    anchorOutputs =     nn.forwardPropogate(feats.anchor);
    positiveOutputs =   nn.forwardPropogate(feats.positive);
    negativeOutputs =   nn.forwardPropogate(feats.negative);

    loss = lossTriplet.computeLoss(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
    
    loss = sum(loss) / tripletCount;
end

% a bit slow; consider speed up
% probably we can reuse already computed outputs but this is not so big
% part of computations
function [ properFeats ] = getProperTriplets( nn, minibatchSize, margin, dataProvider, lossTriplet)

    maxBatchToGenerate = 2048;
    
    persistent batchToGenerate;
    if(isempty(batchToGenerate))
        batchToGenerate = minibatchSize;
    end
    
    properFeats = struct();
    properFeats.anchor = [];
    properFeats.positive = [];
    properFeats.negative = [];
    
    counter = 0;
    
    while(true)
        feats = struct();
        [feats.anchor, feats.positive, feats.negative] = dataProvider.getMinibatch(batchToGenerate);

        anchorOutputs =     nn.forwardPropogate(feats.anchor);
        positiveOutputs =   nn.forwardPropogate(feats.positive);
        negativeOutputs =   nn.forwardPropogate(feats.negative);

        loss = lossTriplet.computeLoss(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
        
        % here we sample rand triplets and find the one that violate the
        % margin. some guys generate rand positive pair and find violating
        % negative. not sure which is better
        
        if(isempty(properFeats.anchor))
            ind = (loss > 0);
            properFeats.anchor =    feats.anchor(ind,:);
            properFeats.positive =  feats.positive(ind,:);
            properFeats.negative =  feats.negative(ind,:);
        else
            ind = (loss > 0);
            
            properFeats.anchor =    vertcat(properFeats.anchor,     feats.anchor(ind,:));
            properFeats.positive =  vertcat(properFeats.positive,   feats.positive(ind,:));
            properFeats.negative =  vertcat(properFeats.negative,   feats.negative(ind,:));
        end
        
        if(size(properFeats.anchor,1) > minibatchSize)
            properFeats.anchor =    properFeats.anchor(1:minibatchSize,:);
            properFeats.positive =  properFeats.positive(1:minibatchSize,:);
            properFeats.negative =  properFeats.negative(1:minibatchSize,:);
        end
        
        if(size(properFeats.anchor,1) == minibatchSize)
            break;
        else
            counter = counter + 1;
            
            if(counter > 2)
                batchToGenerate = batchToGenerate * 2;
                disp(['[getProperTriplets] batchToGenerate increased to ' num2str(batchToGenerate)]);

                if(batchToGenerate >= maxBatchToGenerate)
                    batchToGenerate = maxBatchToGenerate;
                    disp('[getProperTriplets] batchToGenerate to big; truncate with respect to maxBatchToGenerate');
                end
                
            end
        end

    end
            
end


