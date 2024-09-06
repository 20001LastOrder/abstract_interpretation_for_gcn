cd ..
# python verify_experiments.py --type runtime --dataset citeseer --method interval
# python verify_experiments.py --type runtime --dataset citeseer --method poly
# python verify_experiments.py --type runtime --dataset citeseer --method poly_max
# python verify_experiments.py --type runtime --dataset citeseer --method optim

# python verify_experiments.py --type runtime --dataset cora_ml --method interval
# python verify_experiments.py --type runtime --dataset cora_ml --method poly
# python verify_experiments.py --type runtime --dataset cora_ml --method poly_max
# python verify_experiments.py --type runtime --dataset cora_ml --method optim

# python verify_experiments.py --type runtime --dataset pubmed --method interval
# python verify_experiments.py --type runtime --dataset pubmed --method poly
# python verify_experiments.py --type runtime --dataset pubmed --method poly_max
# python verify_experiments.py --type runtime --dataset pubmed --method optim

python verify_experiments.py --type runtime --dataset citeseer --method optim_origin
python verify_experiments.py --type runtime --dataset cora_ml --method optim_origin
python verify_experiments.py --type runtime --dataset pubmed --method optim_origin