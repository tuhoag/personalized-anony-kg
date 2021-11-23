# Protecting Privacy in Knowledge Graphs with Personalized Anonymization

# Initial Setup
To initialize the repository, please install all packages stored in file requirements.txt.

# Data Generation
There are four datasets: Dummy, Email-Eu-Core, Freebase, and Yago in this repository. Dummy is used for our paper example and testing, Email-Eu-Core is a small dataset that we use to test our program. Freebase and Yago are used in our paper. To use a dataset, you must follow three steps: raw knowledge graph generation, k values generation, and distance matrix generation.

Raw Knowledge Graph Generation:
This step imports data from folder /data to outputs/{data_name}/raw such tht our program can run all datasets in the same way. This step is done by running the file generate_raw_kg.py. The implemation of each dataset's imports are in /anonygraph/data whose prefix is the dataset name.

K Values Generation:
To generate k values of users in the dataset, you must use the file generate_k_values.py. There are two options: te, zipfs that are used in our paper.

Distance Matrix Generation: 
Use the file genrate_dist_matrix.py to generate the distance matrix of each dataset.

# Anonymization
The anonymization is done via two steps: clusters generation and knowledge graph generalization. Clusters Generation generate valid clusters while Knowledge Graph Generalization generalizes attributes and relationships.

Clusters Generation:
First, you must use the file generate_raw_clusters.py to use a clustering algorithm to generate clusters. There are three supported clustering algorithm: k-medoids, hdbscan, and vac.

Next, you must use the file anonymize_clusters.py to modify the generated clusters to ensure that all generated clusters in the first step are valid. This is done by running the file anonymize_clusters.py

Knowledge Graph Generalization:
This step is done by running the file anonymize_kg.py. 

# Visualization
To visualize resuts, you can run the file visualize_outputs.py.
