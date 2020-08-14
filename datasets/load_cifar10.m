function [XTrain, YTrain, XTest, YTest] = load_cifar10( dir )
addpath([matlabroot,'/examples/deeplearning_shared/main']);

url = 'https://www.cs.toronto.edu/~kriz/'; % https://www.cs.toronto.edu/~kriz/cifar.html
datasetfile = 'cifar10.mat';

 if( ~exist( 'dir', 'var' ) )
  dir = [fileparts(mfilename('fullpath')),'/'];
 end

 if( dir(end) ~= '/' )
  dir = [dir, '/'];
 end

 
 if( ~isfile( [dir, 'cifar-10-batches-mat/test_batch.mat'] ) )
  disp('Downloading cifar10 dataset. It may take several minitus.');

  filename = 'cifar-10-matlab.tar.gz';
  helperCIFAR10Data.download([url,filename], dir);
  
  %{
  websave([dir,filename], [url,filename]);
  if( Simulink.getFileChecksum([dir,filename]) ~= '70270af85842c9e89bb428ec9976c926' )
   error( 'Download error: cifar-10-matlab.tar.gz' );
  end
  
  gunzip([dir,filename]);
  filename = 'cifar-10-matlab.tar';
  untar( [dir,filename], dir );
  filename = [dir,'cifar-10-matlab.tar'];
  delete( filename );
  %}
  
 end
 [XTrain, YTrain, XTest, YTest] = helperCIFAR10Data.load(dir);

 XTrain = double(XTrain) / 255.0;
 XTest = double(XTest) / 255.0;
end