# DS535 Project

We can use these commands:
```bash
python main.py --model baseline_lightgcn --dataset lastfm --device cuda --cuda 0 (devices you will use)
```

and we will use this command:
```bash
python main.py --model idea_lightgcn --dataset lastfm --device cuda --cuda 0 (devices you will use)
```

We first deal with these files:
1. config/modelconf/{baseline_lightgcn.yml, idea_lightgcn.yml}
2. models/social/{baseline_lightgcn.py, idea_lightgcn.py}
3. (optional) data_utils/datasets_social.py

---
Reference:
https://github.com/HKUDS/SSLRec/tree/main
