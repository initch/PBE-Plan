# CIFAR 10
# Badnets tri1_3x3

# DBA 1x4
python DHBE_train.py --input_dir teacher_badnets_v1_cifar10_resnet18_e_200_DBA_1x4_t0_0_0_n300_results -ps 10 --trigger_name DBA_1x4 --epochs 2000

# DBA_2X4
python DHBE_train.py --input_dir teacher_badnets_v1_cifar10_resnet18_e_200_DBA_2x4_t0_0_0_n300_results -ps 10 --trigger_name DBA_2x4  --epochs 2000 --adjlr 0
python DHBE_train.py --input_dir teacher_badnets_v1_cifar10_resnet18_e_200_DBA_2x4_t0_0_0_n300_results -ps 10 --trigger_name DBA_2x4  --epochs 1200 --adjlr 1

# DBA 1x4 big gap
python DHBE_train.py --input_dir teacher_badnets_v1_cifar10_resnet18_e_200_DBA_1x4_bg_t0_0_0_n300_results -ps 10 --trigger_name DBA_1x4_bg  --epochs 2000 --adjlr 0
python DHBE_train.py --input_dir teacher_badnets_v1_cifar10_resnet18_e_200_DBA_1x4_bg_t0_0_0_n300_results -ps 10 --trigger_name DBA_1x4_bg  --epochs 1200 --adjlr 1

# edge-case
python DHBE_train.py --input_dir neurotoxin_cifar10_resnet18_edgecase_t9_results -ps 10 --backdoor_method edge-case --target_class 9

# semantic
python DHBE_train.py --input_dir semantic_cifar10_resnet18_e100_green-car_t0_n300 --backdoor_method semantic --trigger_name green-car
python DHBE_train.py --input_dir semantic_cifar10_resnet18_e100_racing-stripe_t0_n300 --backdoor_method semantic --trigger_name racing-stripe
python DHBE_train.py --input_dir semantic_cifar10_resnet18_e100_wall_t0_n300 --backdoor_method semantic --trigger_name wall

# Chameleon pixel pattern 1
python DHBE_train.py --input_dir Chameleon_cifar10_resnet18_pixel1_t2_results --backdoor_method chameleon --trigger_name pixel1 --target_class 2 --adjlr 6

# F3BA
python DHBE_train_ff.py --input_dir F3BA_cifar10_resnet18_tri1_3x3_t8 --target_class 8

# howto 
python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_2x2_t0_scale_3 --backdoor_method howto --trigger_name tri1_2x2
python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_3x3_t0_scale_3 --backdoor_method howto --trigger_name tri1_3x3
python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_5x5_t0_scale_3 --backdoor_method howto --trigger_name tri1_5x5

python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_3x3_sacle_2_e270 --backdoor_method howto --trigger_name tri1_3x3
python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_3x3_scale_2_e290 --backdoor_method howto --trigger_name tri1_3x3
python DHBE_train.py --input_dir howto_cifar10_resnet18_tri1_3x3_scale_2_e310 --backdoor_method howto --trigger_name tri1_3x3

# new DBA
python DHBE_train.py --input_dir DBA_cifar10_resnet18_tri2_1x4_gap2_t0 --backdoor_method DBA --trigger_name tri2_1x4
python graft_net_whole.py --input_dir DBA_cifar10_resnet18_tri2_1x4_gap2_t0 --backdoor_method DBA --trigger_name tri2_1x4
python DHBE_train.py --input_dir DBA_cifar10_resnet18_tri2_2x4_gap2_t0 --backdoor_method DBA --trigger_name tri2_2x4
python DHBE_train.py --input_dir DBA_cifar10_resnet18_tri2_1x4_gap24_t0 --backdoor_method DBA --trigger_name tri2_1x4_bg

# neurotoxin
python DHBE_train.py --input_dir neurotoxin_cifar10_resnet18_tri1_2x2_t0 --backdoor_method neurotoxin --trigger_name tri1_2x2
python DHBE_train.py --input_dir neurotoxin_cifar10_resnet18_tri1_3x3_t0 --backdoor_method neurotoxin --trigger_name tri1_3x3
python DHBE_train.py --input_dir neurotoxin_cifar10_resnet18_tri1_5x5_t0 --backdoor_method neurotoxin --trigger_name tri1_5x5


# CIFAR 100
# Badnets
python DHBE_train.py --dataset cifar100 --input_dir teacher_badnets_v1_cifar100_resnet18_e_200_tri1_3x3_t0_0_0_n300_results --backdoor_method Badnets --trigger_name tri1_3x3

# old DBA
python DHBE_train.py --dataset cifar100 --input_dir teacher_badnets_v1_cifar100_resnet18_e_200_DBA_1x4_t0_0_0_n300_results --backdoor_method DBA --trigger_name DBA_1x4
python DHBE_train.py --dataset cifar100 --input_dir teacher_badnets_v1_cifar100_resnet18_e_200_DBA_2x4_t0_0_0_n300_results --backdoor_method DBA --trigger_name DBA_2x4
python DHBE_train.py --dataset cifar100 --input_dir teacher_badnets_v1_cifar100_resnet18_e_200_DBA_1x4_bg_t0_0_0_n300_results --backdoor_method DBA --trigger_name DBA_1x4_bg

# howto
python DHBE_train.py --dataset cifar100 --input_dir howto_cifar100_resnet18_tri1_2x2_t0_5-10_scale_5 --backdoor_method howto --trigger_name tri1_2x2
python DHBE_train.py --dataset cifar100 --input_dir howto_cifar100_resnet18_tri1_3x3_t0_scale_3 --backdoor_method howto --trigger_name tri1_3x3
python DHBE_train.py --dataset cifar100 --input_dir howto_cifar100_resnet18_tri1_5x5_t0_scale_3 --backdoor_method howto --trigger_name tri1_5x5

# new DBA
python DHBE_train.py --dataset cifar100 --input_dir DBA_cifar100_resnet18_tri2_1x4_gap2_t0 --backdoor_method DBA --trigger_name tri2_1x4
python DHBE_train.py --dataset cifar100 --input_dir DBA_cifar100_resnet18_tri2_2x4_gap2_t0 --backdoor_method DBA --trigger_name tri2_2x4
python DHBE_train.py --dataset cifar100 --input_dir DBA_cifar100_resnet18_tri2_1x4_gap24_t0 --backdoor_method DBA --trigger_name tri2_1x4_bg

# neurotoxin
python DHBE_train.py --dataset cifar100 --input_dir neurotoxin_cifar100_resnet18_tri1_2x2_t0 --backdoor_method neurotoxin --trigger_name tri1_2x2
python DHBE_train.py --dataset cifar100 --input_dir neurotoxin_cifar100_resnet18_tri1_3x3_t0 --backdoor_method neurotoxin --trigger_name tri1_3x3
python DHBE_train.py --dataset cifar100 --input_dir neurotoxin_cifar100_resnet18_tri1_5x5_t0 --backdoor_method neurotoxin --trigger_name tri1_5x5
