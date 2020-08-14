function acc = mean_accuracy(YTrue, YPred)

 [m, Yarg] = max(YPred,[],2);
 acc = double( Yarg == int32(YTrue) );
 acc = mean(acc(:));

end