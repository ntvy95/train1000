# train1000

This is my attempt to the coursework of "SCE.I501 Image Recognition", 2020.

## Challenge 

CIFAR-10 (train1000) contains 1,000 training samples (100 samples per class) and 10,000 testing samples. The samples are of very small size (32px x 32px). Therefore, it is a very challenging dataset.

Challenge page: [Train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/).

### Caution

In the original challenge, we must not use the test data for validation. Yet, in the provided framework implementation of the coursework, we were allowed to use the test data as our validation data.

From [the sample code of the original challenge](https://github.com/mastnk/train1000/blob/master/sample_cifar10.py), it seems we must figure out how to utilize 1,000 training samples as validation data, which is even more challenging. Cross-validation seems to be a feasible solution but my limited computational power does not allow me to adopt this approach. Moreover, my coursework did not impose such restriction.

Therefore, in case you are truly dealing with the original challenge, my code is just for reference.

## Structure

./PublicReport.pdf: Detailed report for the choice of my network and my training method.

./net_checkpoint__3520__2020_07_09__15_15_31.mat: My trained model which achieves **57.93%** accuracy. Please read the **Caution** above.

./sample_cifar10.m: Train the network. 

./checkpoint.m: Retrieve the best checkpoint. 

./Evaluate.m: Evaluate a given model.

