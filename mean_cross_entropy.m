function cross_entropy = mean_cross_entropy( YTrue, YPred )

 YTrue_onehot = ind2vec(double(YTrue)')';
 cross_entropy = full(sum( YTrue_onehot .* ( -log( double(YPred) ) ), 2 ));
 cross_entropy = mean(cross_entropy(:));

end