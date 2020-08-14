% follow: https://www.mathworks.com/help/deeplearning/ug/train-residual-network-for-image-classification.html
addpath('datasets');
if ~exist('XTest', 'var')
    [XTrain, YTrain, XTest, YTest] = load_train1000('cifar10');
end
load(['net_checkpoint__3520__2020_07_09__15_15_31.mat'],'net')
[YPred,prob] = classify(net,XTest);
validationAccuracy = 1-mean(YPred ~= YTest);
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YTest,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';