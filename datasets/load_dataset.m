function [XTrain, YTrain, XTest, YTest] = load_dataset( name, dir )

if( ~exist( 'dir', 'var' ) )
 dir = [fileparts(mfilename('fullpath')),'/'];
end

if( dir(end) ~= '/' )
 dir = [dir, '/'];
end

if( strcmpi( name, 'mnist' ) )
  [XTrain, YTrain, XTest, YTest] = load_mnist( dir );

elseif( strcmpi( name, 'cifar10' ) )
  [XTrain, YTrain, XTest, YTest] = load_cifar10( dir );

else
  msg = sptrinf( 'Could not find: %s', name );
  error(msg);
end

end
