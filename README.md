

## Dependency
- numpy (1.14.1)
- pytorch (1.4)



## KO(8,2)

- To train KO(8,2) run default setting by:
    $ python train_KO_m2.py --gpu 0 --m 8 --enc_train_snr 0 --dec_train_snr -2 --batch_size 100000


- To test KO(8,2) :
    python test_KO_m2.py --gpu 0 --m 8 --enc_train_snr 0 --dec_train_snr -2 --batch_size 100000

## KO(9,2)

- To train KO(9,2) run default setting by:
    $ python train_KO_m2.py --gpu 0 --m 9 --enc_train_snr -2 --dec_train_snr -4 --batch_size 50000


- To test KO(9,2) :
    python test_KO_m2.py --gpu 0 --m 9 --enc_train_snr -2 --dec_train_snr -4 --batch_size 50000


## KO(6,1)

- To train KO(6,1) run default setting by:
    $ python train_KO_m1_dumer.py --gpu 0 --m 6


- To test KO(6,1) :
    python test_KO_m1_dumer.py --gpu 0 --m 6


## KO(6,1) with MAP decoding

- To train KO(6,1) with MAP decoding run default setting by:
    $ python train_KO_m1_map.py --gpu 0 --m 6


- To test KO(6,1) with MAP decoding:
    python test_KO_m1_map.py --gpu 0 --m 6


## Polar(64,7) 

- To train Polar(64,7) run default setting by:
    $ python train_Polar_m6k7.py --gpu 0 


- To test Polar(64,7):
    python test_Polar_m6k7.py --gpu 0