
##Stochastic Variational Inference for Latent Dirichlet Allocation

Code structure from the OnlineVB code provided by Matthew D. Hoffman (mdhoffma@cs.princeton.edu) and the algorithm is as described in Hoffman's paper below

Based on the following papers:
- [Latent Dirichlet Allocation](https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf) by David M. Blei, Andrew Y. Ng and Michael I. Jordan
- [Stochastic Variational Inference](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf) by Matthew D. Hoffman, David M. Blei, Chong Wang and John Paisley

###Also aiming to implement SVI for HDP as described in the second paper above, work in progress


###How to Use
See 'Help' using
```python stochastic_lda.py -h```

You will need:
- A file [dictionary.csv] containing your vocabular
- A file [doclist.txt] containing the list of documents in the directory that you want to sample from
- At the moment your documents can be just a normal txt file, no pre-processing required

For classwork, work in progress...

- [x] Basic initial implementation
- [x] Debug for common corpus
- [x] Support Command-Line Usage for user-defined test mode and normal mode
- [x] Run on own data
- [ ] Implement HDP
