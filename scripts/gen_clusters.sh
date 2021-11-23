DATA = email
GEN_ARGS = 0.002,0.005,0.001,2.5

CALGO=km

python run_generate_clusters.py --data=email --sample=-1 --k_gen=zipf --k_args=2,5,1,0.6 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=cz --n_gens=3 --log=i --workers=2

python run_generate_clusters.py --data=email --sample=-1 --k_gen=zipf --k_args=2,10,1,0.6 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=n --n_gens=3 --log=i --workers=2

python run_generate_clusters.py --data=email --sample=-1 --k_gen=zipf --k_args=2,10,1,1 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=n --n_gens=3 --log=i --workers=2

python run_generate_clusters.py --data=email --sample=-1 --k_gen=zipf --k_args=2,20,2,0.6 --calgo=km --enforcer=kp --max_cost_list=3,3.5 --max_dist_list=0,0.25,0.5,0.75,1 --handler=n --n_gens=3 --log=i --workers=2
