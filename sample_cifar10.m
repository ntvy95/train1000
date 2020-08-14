reqToolboxes = {'Deep Learning Toolbox'};
if( ~checkToolboxes(reqToolboxes) )
 msg = 'It requires:';
 for i=1:numel(reqToolboxes)
  msg = [msg, reqToolboxes{i}, ', ' ];
 end
 msg = [msg, 'Please install these toolboxes.'];
 error(msg);
end

% help
% https://mathworks.com/help/deeplearning/ref/classify.html

addpath('datasets');

if( ~exist( 'train1000', 'var' ) )
    train1000 = true;
end
if 1
    if ~exist('TrainData.mat')
        if( train1000 )
         [XTrain, YTrain, XTest, YTest] = load_train1000('cifar10');
        else
         [XTrain, YTrain, XTest, YTest] = load_dataset('cifar10');
        end
            for i = 1:1000
                Xs = reshape(jitterColorHSV(XTrain(:,:,:,i),'Saturation',[-0.4 -0.1]), 32, 32, 3, 1); 
                Xh = reshape(jitterColorHSV(XTrain(:,:,:,i),'Hue',[0.05 0.15]), 32, 32, 3, 1); 
                Xb = reshape(jitterColorHSV(XTrain(:,:,:,i),'Brightness',[-0.3 -0.1]), 32, 32, 3, 1); 
                Xn = reshape(imnoise(XTrain(:,:,:,i), 'gaussian'), 32, 32, 3, 1); 
                Xbl = reshape(imgaussfilt(XTrain(:,:,:,i), 1+5*rand), 32, 32, 3, 1); 
                Xshb = reshape(jitterColorHSV(XTrain(:,:,:,i),'Saturation',[-0.4 -0.1],...
                    'Hue',[0.05 0.15],'Brightness',[-0.3 -0.1]), 32, 32, 3, 1); 
                Xshbn = reshape(imnoise(Xshb(:,:,:,1), 'gaussian'), 32, 32, 3, 1);
                Xshbnl = reshape(imgaussfilt(Xshb(:,:,:,1), 1+5*rand), 32, 32, 3, 1);
                XTrain = cat(4, XTrain, Xs, Xh, Xb, Xshb, Xn, Xbl, Xshbnl);
                YTrain = cat(1, YTrain, YTrain(i), YTrain(i), YTrain(i), YTrain(i), YTrain(i), YTrain(i), YTrain(i));
            end
        save('TrainData.mat', 'XTrain', 'YTrain');
    else
        if 0
            [XTrain, YTrain, XTest, YTest] = load_train1000('cifar10');
            load('TrainData.mat')
        end
    end
else
    [XTrain, YTrain, XTest, YTest] = load_train1000('cifar10');
end

% follow: https://www.mathworks.com/help/deeplearning/ug/image-augmentation-using-image-processing-toolbox.html

imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20],...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXShear',[-0.017 0.017],...
    'RandYShear',[-0.017 0.017]);
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');

% follow: 
% https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html
% https://github.com/mastnk/train1000/blob/master/sample_cifar10.py

nb_classes = 10;

lgraph = layerGraph([ ...
    imageInputLayer([32 32 3], 'Name', 'input')
    
    convolution2dLayer(3, 32, 'padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv_1')
    reluLayer('Name','relu_1')
    convolution2dLayer(3, 32, 'padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv_2')
    reluLayer('Name','relu_2')

    averagePooling2dLayer(2,'Stride',2, 'Name', 'pool_1')
    
    dropoutLayer(0.25, 'Name', 'drop_1')
    ]);

lgraph = addLayers(lgraph, [ ...
    convolution2dLayer(2, 32, 'stride', 2, 'padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv_1_1')
    reluLayer('Name','relu_1_1')
    ]);
    
lgraph = addLayers(lgraph,[convolution2dLayer(3, 64, 'padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv_3')
    reluLayer('Name','relu_3')
    convolution2dLayer(3, 64, 'padding', 'same', 'WeightsInitializer', 'he', 'Name', 'conv_4')
    reluLayer('Name','relu_4')

    averagePooling2dLayer(2,'Stride',2, 'Name', 'pool_2')
    
    dropoutLayer(0.25, 'Name', 'drop_2')
   ]);

lgraph = addLayers(lgraph,[fullyConnectedLayer(128,'Name','fc_1')    
    reluLayer('Name','relu_5')

    dropoutLayer(0.5, 'Name', 'drop_3')
    
    fullyConnectedLayer(nb_classes,'Name','fc_2')    
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')]);
lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name','concat_1'));
lgraph = connectLayers(lgraph,'drop_1','conv_3');
lgraph = connectLayers(lgraph,'drop_1','conv_1_1');
lgraph = connectLayers(lgraph,'relu_1_1','concat_1/in1');
lgraph = connectLayers(lgraph,'drop_2','concat_1/in2');
lgraph = connectLayers(lgraph,'concat_1','fc_1');

options = trainingOptions('adam', ...
    'Shuffle','every-epoch', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 100, ...
    'ValidationData',{XTest, YTest}, ...
    'ValidationFrequency', 10, ...
    'Plots','training-progress', ...
    'CheckpointPath', '/MATLAB Drive/checkpoint/');

%load('net_checkpoint__3520__2020_07_09__15_15_31.mat', 'net');
%lgraph = layerGraph(net);

net = trainNetwork(augimdsTrain,lgraph,options);

YPred = predict(net,XTrain);
acc = mean_accuracy( YTrain, YPred );
ce = mean_cross_entropy( YTrain, YPred );
fprintf( 'Train mean accuracy: %g\n', acc );
fprintf( 'Train mean cross entropy: %g\n\n', ce );

YPred = predict(net,XTest);
acc = mean_accuracy( YTest, YPred );
ce = mean_cross_entropy( YTest, YPred );
fprintf( 'Test mean accuracy: %g\n', acc );
fprintf( 'Test mean cross entropy: %g\n\n', ce );

if( train1000 )
 disp( '********** ********** ********** **********' );
 disp( '* It was trained with just 1000 samples.' );
 disp( '* Please visit train with 1000 project page: <a href="http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/">http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/</a>' );
 disp( '* ' );
 disp( '* If you want to train with full size of training data, please run as follow:' );
 disp( '* >> train1000 = false; sample_cifar10;' );
 disp( '********** ********** ********** **********' );
end
