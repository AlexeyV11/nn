function [ output_args ] = trainTripletLossNetwork( nn, epochs, minibatchSize, margin, input_features, output_labels)
    
    [~,labels] = max(output_labels');
    
    features = cell(1,max(labels));
    for i=1:numel(features)
        features{i} = {};
    end
    
    for i=1:numel(labels)
        features{labels(i)}{end+1} = input_features(i,:);
    end
    
    lossTriplet = LossTriplet(margin);
    
    for epoch = 1:epochs
        itersCount = floor(size(output_labels,1)/minibatchSize);
        
        lossEpoch = [];
        
        for iters = 1:itersCount 
            
            feats = struct();
            [feats.anchor, feats.positive, feats.negative] = generate_triplets(features, minibatchSize);
    
            anchorOutputs = nn.forwardPropogate(feats.anchor);
            positiveOutputs = nn.forwardPropogate(feats.positive);
            negativeOutputs = nn.forwardPropogate(feats.negative);
            
            
            loss = lossTriplet.computeLoss(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
        
            [anchor_derivative, positive_derivative, negative_derivative] = lossTriplet.computeDerivative(anchorOutputs{end}, positiveOutputs{end}, negativeOutputs{end});
        
            

            
            %nn.backPropagate(output_train_batch, lossLayer.computeDerivative(output_train_batch{end}, answers));
        end

        disp(['epoch : ' num2str(epoch) ' loss : ' num2str(sum(lossEpoch) / numel(lossEpoch))]);
    end
end


function [anchor_feats, positive_feats, negative_feats] = generate_triplets(features, triplets_count)
    
    debug_log = 0;
    
    count = 0;
    feat_shift = [];
    feat_count = [];
    
    feat_for_person_min = 100000000000;
    feat_for_person_max = -1;
    
    for i = 1:numel(features)
        feat_shift(end+1) = count;
        feat_count(end+1) = numel(features{i});
        count = count + numel(features{i});
        
        feat_for_person_min = min(feat_for_person_min, numel(features{i}));
        feat_for_person_max = max(feat_for_person_max, numel(features{i}));
        
    end
    
    positive_person_index = randi([1 numel(features)],1,triplets_count);
    
    negative_person_index = positive_person_index + randi([0 numel(features) - 2],1,triplets_count);
    negative_person_index = rem(negative_person_index, numel(features))+1;
    
    if(debug_log)
        disp(['person index check ' num2str(sum(positive_person_index == negative_person_index)==0)]);
    end
    
    positive_sizes = feat_count(positive_person_index);
    positive_shifts = feat_shift(positive_person_index);
    positive_arr_index_1 = positive_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), positive_sizes) + 1;
    positive_arr_index_2 = positive_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), positive_sizes) + 1;
    
    if(debug_log)

        disp(['arr index check 1 ' num2str(sum(positive_arr_index_1 > feat_shift(positive_person_index)) == triplets_count)]);
        disp(['arr index check 2 ' num2str(sum(positive_arr_index_1 <= feat_shift(positive_person_index)+feat_count(positive_person_index)) == triplets_count)]);
        disp(['arr index check 3 ' num2str(sum(positive_arr_index_2 > feat_shift(positive_person_index)) == triplets_count)]);
        disp(['arr index check 4 ' num2str(sum(positive_arr_index_2 <= feat_shift(positive_person_index)+feat_count(positive_person_index)) == triplets_count)]);
    end
    
    negative_sizes = feat_count(negative_person_index);
    negative_shifts = feat_shift(negative_person_index);
    negative_arr_index = negative_shifts + rem(randi([0 feat_for_person_max],1,triplets_count), negative_sizes) + 1;
    
    if(debug_log)
        disp(['arr index check 5 ' num2str(sum(negative_arr_index > feat_shift(negative_person_index)) == triplets_count)]);
        disp(['arr index check 6 ' num2str(sum(negative_arr_index <= feat_shift(negative_person_index)+feat_count(negative_person_index)) == triplets_count)]);
    end
       
    arr = zeros(count, 784);
    
    ind = 1;
    for i = 1:numel(features)
        for j = 1:numel(features{i})
            feat = features{i}{j};
            arr(ind,:) = feat(1,:);
            
            ind = ind + 1;
        end
    end
    triplets = [positive_arr_index_1' positive_arr_index_2' negative_arr_index'];
    
    inds = positive_arr_index_1' ~= positive_arr_index_2';
    triplets = triplets(inds,:);
    
    anchor_feats = arr(triplets(:,1),:);
    positive_feats = arr(triplets(:,2),:);
    negative_feats = arr(triplets(:,3),:);
end