function [XTrain, YTrain, XTest, YTest] = load_train1000( name, dir )

if( ~exist( 'dir', 'var' ) )
 dir = [fileparts(mfilename('fullpath')),'/'];
end

if( dir(end) ~= '/' )
 dir = [dir, '/'];
end

if( strcmpi( name, 'mnist' ) )
  [XTrain0, YTrain0, XTest, YTest] = load_mnist( dir );
  [XTrain, YTrain] = extract(XTrain0, YTrain0, 10, 100);

elseif( strcmpi( name, 'cifar10' ) )
  [XTrain0, YTrain0, XTest, YTest] = load_cifar10( dir );
  [XTrain, YTrain] = extract(XTrain0, YTrain0, 10, 100);
  
else
  msg = sptrinf( 'Could not find: %s', name );
  error(msg);
end

end

function [X, Y] = extract(X0, Y0, nb_classes, nb_per_class)
 s = size(X0);
 X = X0(:,:,:,1:nb_classes*nb_per_class);
 Y = Y0(1:nb_classes*nb_per_class,:);
 
 k = 1;
 n = zeros(nb_classes,1);
 for i=1:size(X0,1)
  c = int32(Y0(i,1));
  if( n(c) < nb_classes )
   X(:,:,:,k) = X0(:,:,:,i);
   Y(k,:) = Y0(i,:);
   n(c) = n(c) + 1;
   k = k + 1;
   if( k > nb_classes*nb_per_class )
    break;
   end
  end
 end
end