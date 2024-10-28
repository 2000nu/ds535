# DS535 Project

We can use these commands:
```bash
python main.py --model mhcn --dataset yelp --device cuda --cuda 0 (devices you will use)
python main.py --model baseline_mhcn --dataset yelp --device cuda --cuda 0 (devices you will use)
```

and we will use this command:
```bash
python main.py --model idea_mhcn --dataset yelp --device cuda --cuda 0 (devices you will use)
```

We first deal with these files:
1. config/modelconf/{baseline_mhcn.yml, idea_mhcn.yml}
2. models/social/{baseline_mhcn.py, idea_mhcn.py}
3. (optional) data_utils/datasets_social.py

I put "TODO" in models/social/idea_mhcn.py, so to-do lists are:
1. complete '_calculate_homophily_ratios'
2. complete '_rewire_social_graph'
3. complete 'cal_cl_loss'

and I hope to fill in these files somedays:
1. config/modelconf/{baseline_lightgcn.yml, idea_lightgcn.yml}
2. models/social/{baseline_lightgcn.py, idea_lightgcn.py}

---
Reference:
https://github.com/HKUDS/SSLRec/tree/main
