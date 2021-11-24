# Protecting Privacy in Knowledge Graphs with Personalized Anonymization

# Initial Setup
To initialize the repository, please install all packages stored in file requirements.txt.

# Data Generation
There are four datasets: Dummy, Email-Eu-Core, Freebase, and Yago in this repository. Dummy is used for our paper example and testing, Email-Eu-Core is a small dataset that we use to test our program. Freebase and Yago are used in our paper. To use a dataset, you must follow three steps: raw knowledge graph generation, k values generation, and distance matrix generation.

## Raw Knowledge Graph Generation:
This step imports data from folder /data to outputs/{data_name}/raw such tht our program can run all datasets in the same way. This step is done by running the file generate_raw_kg.py. The implemation of each dataset's imports are in /anonygraph/data whose prefix is the dataset name.

For example, to generate the original knowlege graph of Freebase, you can run the following code:
```
python generate_raw_kg.py --data=freebase
```

## K Values Generation:
To generate k values of users in the dataset, you must use the file generate_k_values.py. There are two options: te, zipfs that are used in our paper. The following code generate k values for Freebase's users based on zipf and te setting:
```
python generate_k_values.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --gen_n=0
python generate_k_values.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --gen_n=1
python generate_k_values.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --gen_n=2
python generate_k_values.py --data=freebase --gen=te --gen_args=2,5,1
```
Here, 2,5,1 is the range start from 2 to 5 with the increment of 1. The last parameter of zipf setting is zipf distribution parameter, which is 2 in the above code. The first three commands generate k values three times.

## Distance Matrix Generation: 
Use the file genrate_dist_matrix.py to generate the distance matrix of each dataset. You can use the following code to generate distance matrix of Freebase:
```
python generate_dist_matrix.py --data=freebase --gen=te --gen_args=2,5,1,2 --workers=1
python generate_dist_matrix.py --data=freebase --gen=te --gen_args=2,5,1 --workers=1
```
Here, you can speed up the performance of this step by passing the number of processors that you want to use for this step. In the above code, the program only uses one processor.

# Anonymization
The anonymization is done via two steps: clusters generation and knowledge graph generalization. Clusters Generation generate valid clusters while Knowledge Graph Generalization generalizes attributes and relationships.

## Clusters Generation:
First, you must use the file generate_raw_clusters.py to use a clustering algorithm to generate clusters. There are three supported clustering algorithm: k-medoids, hdbscan, and vac. The following code generates raw clusters of Freebase's users from three clustering algorithm hdbscan, k-medoids, and vac. With hdbscan and k-medoids, you must pass a parameter to calculate how to specify the number of generated clusters and minimum size of these clusters. VAC can automatically exploit users' k values and you do not need the specify any andditional parameters.

```
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=hdbscan --calgo_args=max
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=hdbscan --calgo_args=min
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=hdbscan --calgo_args=mean
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=km --calgo_args=max
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=km --calgo_args=min
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=km --calgo_args=mean
python generate_raw_clusters.py --data=freebase --gen=zipf --gen_args=2,5,1,2 --calgo=vac 
```
Here, you can pass ```gen_n```(default=0) to specify this generation relies on which k values that are generated.

Next, you must use the file anonymize_clusters.py to modify the generated clusters to ensure that all generated clusters in the first step are valid. This is done by running the file anonymize_clusters.py. The following code generates valid clusters by using two enforcers: Small Removal (aka Invalid Removal) and Merge Split (ms). Merge Split requires a threshold to calculate the maximum distance between users and valid clusters.
```
python anonymize_clusters.py --data=freebase --gen=te --gen_args=2,5,1,2 --calgo=km --calgo_args=max --enforcer=sr
python anonymize_clusters.py --data=freebase --gen=te --gen_args=2,5,1,2 --calgo=km --calgo_args=max --enforcer=ms --enforcer_args=1
```
Here, the second command pass the threshold as ```1``` to the Merge Split enforcer.

## Knowledge Graph Generalization:
This step is done by running the file anonymize_kg.py. The following code generates the anonymized knowledge graph from the generated valid clusters
```
python anonymize_kg.py --data=freebase --gen=te --gen_args=2,5,1 --calgo=km --calgo_args=max --enforcer=ms --enforcer_args=1
```

# Visualization
To visualize resuts, you can run the file visualize_outputs.py. The following code collects generated knowledge graphs, extracts their quality, stores in csv files, and visualizes them:
```
python visualize_outputs.py --data_list=freebase,yago --refresh=y,y --src_type=graphs --exp_names=vac,ms,compare --workers=1
```
This visualization firstly collects anonymized knowledge graphs. Secondly, it aggregates results of anonymized knowledge graphs generated with the same setting but different generation ```gen_n```. In particular the above code refresh data from both steps. If you finished the first and only one to aggregate the data again, you can pass ```refresh=n,y```. If you only want to visualize, ```refresh=n,n```.
