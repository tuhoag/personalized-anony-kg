python run_generate_clusters.py --data=email --sample=-1 --k_gen=zipf --k_args=2,5,1,0.6 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=n --n_gens=3 --log=i --workers=2

python run_generate_anonymized_clusters.py --data=email --sample=-1 --k_gen=od --k_args=0.01,10 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2

python run_generate_anonymized_clusters.py --data=email --sample=-1 --k_gen=od --k_args=0.05,10 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2

python run_generate_anonymized_clusters.py --data=email --sample=-1 --k_gen=od --k_args=0.1,10 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2

python run_generate_anonymized_clusters.py --data=email --sample=-1 --k_gen=od --k_args=0.15,10 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2

python run_generate_anonymized_clusters.py --data=email --sample=-1 --k_gen=od --k_args=0.2,10 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2