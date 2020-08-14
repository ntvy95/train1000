addpath('datasets');
listing = dir('checkpoint');

if ~exist('XTest', 'var')
    [XTrain, YTrain, XTest, YTest] = load_train1000('cifar10');
end

best_acc = 0;
best_ce = inf;
best_i = 0;

for i=3:length(listing)
    load(['/MATLAB Drive/checkpoint/',listing(i).name],'net')
    YPred = predict(net,XTest);
    acc = mean_accuracy( YTest, YPred );
    ce = mean_cross_entropy( YTest, YPred );
    fprintf( 'Model: %s\n', listing(i).name );
    fprintf( "Test mean accuracy: %g\n", acc );
    fprintf( "Test mean cross entropy: %g\n\n", ce );
    if acc > best_acc
        best_acc = acc;
        best_ce = ce;
        best_i = i;
    end
end

disp( '********** ********** ********** **********' );
fprintf( 'Best model: %s\n', listing(best_i).name );
fprintf( "Best model's test mean accuracy: %g\n", best_acc );
fprintf( "Best model's test mean cross entropy: %g\n\n", best_ce );

% 'net_checkpoint__3520__2020_07_09__15_15_31.mat', 57.93%, 1.76408