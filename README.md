# Qwen3.8 Flash Next NVFP4 sur Vast.ai

Image de déploiement pour
[`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4),
optimisée pour deux GPU Blackwell de 96 Go et une API vLLM compatible OpenAI.

L’image publiée est :

```text
ghcr.io/cluster2600/qwen38-flash-next-vast:latest
```

Référence immuable du premier build validé :

```text
ghcr.io/cluster2600/qwen38-flash-next-vast@sha256:daa7d2ac790d70500f304cad6c38c3f0e8dfbf76c14f20a5a07b5a113e4f006f
```

Elle ne contient ni les poids du modèle, ni jeton Hugging Face, ni autre secret.
Le dépôt Hugging Face étant soumis à acceptation, `HF_TOKEN` doit être fourni à
l’instance par l’utilisateur. Les poids représentent environ 189 Go ; prévoir au
moins 300 Go de disque local.

## Profil matériel

- 2× RTX PRO 6000 Blackwell 96 Go, compute capability 12.0 ;
- au moins 128 Go de RAM hôte, dont environ 64 Go libres pour l’offload PLE ;
- PCIe 5.0 recommandé ;
- 300 Go de disque au minimum ;
- Docker avec accès aux deux GPU.

Le démarrage refuse les GPU antérieurs à Blackwell et toute configuration qui
n’expose pas exactement deux GPU.

## Démarrage

Le conteneur lit les variables suivantes :

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `HF_TOKEN` | requis | accès au dépôt Hugging Face soumis à acceptation |
| `HF_HOME` | `/workspace/huggingface` | cache persistant des poids |
| `MAX_MODEL_LEN` | `32768` | contexte maximal exposé à Hermes |
| `MAX_NUM_SEQS` | `8` | concurrence, multiple de 4 |
| `GPU_MEMORY_UTILIZATION` | `0.95` | budget VRAM vLLM |
| `SPECULATIVE_TOKENS` | `3` | profondeur MTP |
| `NCCL_P2P_DISABLE` | `1` | évite le deadlock NCCL des doubles Blackwell sous IOMMU/ACS |

Exécution directe :

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN \
  -v /chemin/vers/le/cache:/workspace/huggingface \
  ghcr.io/cluster2600/qwen38-flash-next-vast:latest
```

Sur Vast.ai en mode SSH, utiliser le script d’arrière-plan comme commande de
démarrage :

```bash
/opt/qwen/start-background.sh
```

Les logs sont écrits dans `/workspace/qwen-vllm/server.log`.

## Accès sécurisé et Hermes

vLLM écoute uniquement sur `127.0.0.1:8000`. Depuis le Mac :

```bash
ssh -N -L 18000:127.0.0.1:8000 root@HOTE_VAST -p PORT_SSH
```

Hermes utilisera ensuite `http://127.0.0.1:18000/v1`. Aucun port vLLM public
non authentifié n’est nécessaire.

## Validation des 60 tokens/s

Après que `/opt/qwen/healthcheck.py` indique le modèle chargé :

```bash
/opt/qwen/benchmark.py
```

Le benchmark chauffe d’abord le chemin de décodage, génère ensuite 512 tokens
en flux unique et échoue si la moyenne mesurée est inférieure à 60 tokens/s.
Le seuil peut être changé avec `MIN_TOKENS_PER_SECOND`.

Un déploiement documenté sur 2× RTX PRO 6000 Blackwell, TP=2, PLE en RAM et
MTP=3 mesure environ 161 tokens/s en flux unique. Ce résultat est une référence,
pas une garantie pour une offre Vast donnée : la machine louée reste validée par
le benchmark avant la configuration de Hermes.

## Choix techniques

- image vLLM temporaire `vllm/vllm-openai:qwen38-flash-next`, nécessaire à
  l’architecture récente `qwen4_exp` ;
- offload en RAM de la table PLE ;
- correctif ciblé du résolveur PLE pour le checkpoint hybride NVFP4/FP8 ;
- normalisation ciblée de `qwen_sparse_attention` vers le chemin QSA appelé
  `full_attention` dans l’image vLLM day-zero ;
- CUDA Graphs sur le décodage et MTP spéculatif ;
- custom all-reduce désactivé sur SM120 ;
- P2P NCCL désactivé par défaut pour tolérer les hôtes Vast avec IOMMU/ACS ;
- API limitée à la boucle locale.

Références :

- [fiche du modèle OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4) ;
- [déploiement et benchmarks 2× RTX PRO 6000](https://github.com/SirTificate/qwen38-flash-next-2x-rtx6000-sm120) ;
- [image de base Vast.ai et accès SSH](https://github.com/vast-ai/base-image).

## Limites

Le modèle est désaligné et sa couche de refus a été retirée. Son utilisation
doit rester conforme au droit applicable et inclure les garde-fous nécessaires.
Les deux correctifs vLLM sont volontairement stricts : le build échoue si la
structure des fichiers ciblés change afin d’éviter d’appliquer silencieusement
une modification incompatible.
