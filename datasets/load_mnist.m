function [XTrain, YTrain, XTest, YTest] = load_mnist( dir )

url = 'http://yann.lecun.com/exdb/mnist/';
datasetfile = 'mnist.mat';


 if( ~exist( 'dir', 'var' ) )
  dir = [fileparts(mfilename('fullpath')),'/'];
 end

 if( dir(end) ~= '/' )
  dir = [dir, '/'];
 end

 if( isfile( [dir, datasetfile] ) )
  load( [dir, datasetfile] );
 else
  fprintf( 'Downloading mnist dataset. It may takes several minitus.\n' );
     
  
  filename = 'train-images-idx3-ubyte';
  websave([dir,filename,'.gz'], [url,filename, '.gz']);
  gunzip([dir,filename, '.gz']);
  XTrain = readMnistX([dir,filename]);
  filename = [ dir, filename, '*' ];
  delete(filename);
  
  filename = 'train-labels-idx1-ubyte';
  websave([dir,filename,'.gz'], [url,filename, '.gz']);
  gunzip([dir,filename, '.gz']);
  YTrain = readMnistY([dir,filename]);
  filename = [ dir, filename, '*' ];
  delete(filename);
  
  filename = 't10k-images-idx3-ubyte';
  websave([dir,filename,'.gz'], [url,filename, '.gz']);
  gunzip([dir,filename, '.gz']);
  XTest = readMnistX([dir,filename]);
  filename = [ dir, filename, '*' ];
  delete(filename);

  filename = 't10k-labels-idx1-ubyte';
  websave([dir,filename,'.gz'], [url,filename, '.gz']);
  gunzip([dir,filename, '.gz']);
  YTest = readMnistY([dir,filename]);
  filename = [ dir, filename, '*' ];
  delete(filename);
  
  save([dir,datasetfile], 'XTrain', 'YTrain', 'XTest', 'YTest');
 end
 
 XTrain = double(XTrain) / 255.0;
 YTrain = categorical(YTrain);
 XTest = double(XTest) / 255.0;
 YTest = categorical(YTest);

end

function X = readMnistX( filename )
    fid = fopen(filename, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2051
        error('Invalid image file header');
    end
    count = fread(fid, 1, 'int32');
    h = fread(fid, 1, 'int32');
    w = fread(fid, 1, 'int32');
    
    X = fread(fid, h*w*count, 'uint8');
    X = reshape( X, [h,w,1,count] );
    X = permute( X, [2,1,3,4] );
    
    X = uint8(X);
    fclose(fid);
end

function Y = readMnistY( filename )
    fid = fopen(filename, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2049
        error('Invalid image file header');
    end
    count = fread(fid, 1, 'int32');
    
    Y = fread(fid, count, 'uint8');
    Y = reshape( Y, [count,1] );
    Y = uint8(Y);
    
    fclose(fid);
end