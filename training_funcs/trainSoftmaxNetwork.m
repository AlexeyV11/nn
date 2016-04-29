function [ output_args ] = trainSoftmaxNetwork( nn, epochs, minibatchSize, input_features, output_labels)
    lossLayer = LossSoftmax(size(input_features,2));
    
    for epoch = 1:epochs
        itersCount = floor(size(output_labels,1)/minibatchSize);
        
        for iters = 1:itersCount 
            samples = input_features((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            answers = output_labels((iters-1)*minibatchSize+1:iters*minibatchSize,:);
            
            output_train_batch = nn.forwardPropogate(samples);
            loss = lossLayer.computeLoss(output_train_batch{end}, answers);
            
            nn.backPropagate(output_train_batch, lossLayer.computeDerivative(output_train_batch{end}, answers));
        end
    end
end

