# PL9 · Diffusion & Multimodality — guia para o Claude

## Contexto
- UC: Inteligência Artificial Generativa (MIA, Universidade do Minho), 2026/27.
- Aula teórica: `T8.pdf` (Diffusion & Multimodality). Prática: `PL9.ipynb`.
- Entregável: notebook executado de cima a baixo, com os gráficos das Partes 1–4, os resultados de CLIP da Parte 5 e respostas curtas a todas as perguntas ❓ Q1–Q7 + relatório da Parte 7.
- O notebook está pensado para **Google Colab** (usa `!pip`, pasta `imagens/` como plano B). Localmente, o Python com bibliotecas é `~/miniconda3/bin/python`.

## Como me ajudar (modo tutor)
- Responder sempre em **português de Portugal**.
- Objetivo é **aprender** para o exame, não só entregar. Por defeito:
  1. Explicar o conceito ligando-o ao slide do T8 correspondente.
  2. Dar uma pista / pseudo-código e deixar-me tentar escrever o `# TODO`.
  3. Se eu pedir explicitamente ("dá-me o código", "resolve"), dar a solução completa, comentada.
- Nas perguntas ❓, ajudar-me a raciocinar e rever a minha resposta; não escrever a resposta final por mim a não ser que eu peça.
- Não alterar células que não são `# TODO` sem eu pedir. Editar o `.ipynb` com a ferramenta NotebookEdit.
- Não fazer commit/push sem eu pedir.

## Mapa notebook ↔ slides
| Parte | O que se faz | Slides T8 |
|---|---|---|
| 0 | Setup, dados "two moons" normalizados | — |
| 1 | `cosine_schedule`, `q_sample`, Q1 (cosine vs linear) | 5–6 (forward process) |
| 2 | `diffusion_loss` (MSE no ruído), Q2 | 6 (training) |
| 3 | DDPM (dado), `ddim_sample` (TODO), timing Q3 | 7–8 (quality–speed dial, DDIM) |
| 4 | CFG: `eps = eps_u + w*(eps_c - eps_u)`, Q4 | 9–10 (guidance) |
| 5 | CLIP zero-shot (`zero_shot`), templates de prompt Q5, retrieval Q6 | 15 (CLIP) |
| 6 | Sondar CLIP: contagem, negação, relações, Q7 | 16–17, 19 (VLM, "can CLIP count?") |
| 7 | Relatório | 25 (cinco ideias-chave) |
| ★ | B1 Stable Diffusion (GPU), B2 HyDE com CLIP, B3 interpolação | 11, 13 |

## Armadilhas técnicas conhecidas (verificar quando algo falha)
- **cosine_schedule** tem de devolver *tensores torch* (o DDPM faz `a.sqrt()`), com tamanho `T`. `alphas[0] = abar[0]`, `alphas[1:] = abar[1:]/abar[:-1]`; fazer clamp dos betas a `[0, 0.999]` e recalcular `abar = cumprod(1 - betas)`.
- **q_sample**: `abar[t]` tem forma `(B,)` → fazer `.unsqueeze(1)` (ou `[:, None]`) para fazer broadcast com `x0` de forma `(B, 2)`. Mover `abar` para `x0.device` antes de indexar.
- **Schedule linear (Q1)**: os betas clássicos `1e-4…0.02` são para T=1000; com T=200 multiplicar por `1000/T`, senão `abar[-1]` não chega a ≈0.
- **diffusion_loss**: `((model(xt, t) - noise)**2).mean()`; passar `c` só se não for `None`.
- **DDIM**: fazer clamp de `x0` a `[-3, 3]`; sem isso o primeiro passo explode (`sqrt(abar[T-1]) ≈ 2.5e-4`).
- **zero_shot**: `logits = clip_model.logit_scale.exp() * img @ txt.T` → `softmax(dim=-1)`.
- Na Parte 5, se a Wikimedia bloquear os downloads, usar o plano B (pasta `imagens/`, nome do ficheiro = label).
- Os resultados variam ligeiramente entre execuções; existe `torch.manual_seed(0)` na Parte 0.
