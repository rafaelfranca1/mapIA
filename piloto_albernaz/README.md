# Piloto Albernaz (19/11 a 10/12)

Curadoria manual nesta semana. O notebook (ResNet-50 pré-treinada como extratora e similaridade cosseno) fica para 26/11–02/12.

## Onde colocar cada arquivo

| Pasta | O que entra |
| --- | --- |
| `mapas_inteiros/referencia/` | Cartas confirmadas de Albernaz I |
| `mapas_inteiros/consulta/` | Carta reservada, Pernambuco ca. 1630 |
| `recortes/escrita/` | Topônimos e inscrições manuscritas correntes |
| `recortes/contorno/` | Margens, recifes e linhas costeiras |
| `recortes/titulo/` | Títulos ornamentais, se forem recortados |

`catalogo_cartas.csv` e `tabela_recortes.csv` estão com as linhas de exemplo da professora. Troque arquivo, fonte e estado da imagem pelos dados reais antes de tratar o CSV como catálogo definitivo.

## Recortes

- Duas categorias no experimento: `escrita` e `contorno`. Título ornamental entra em `titulo`, e só entra na comparação se as cartas de referência tiverem título equivalente.
- Corte fora rasgos, fita adesiva, carimbo de biblioteca e anotação moderna a lápis.
- Nome do arquivo: `<id_mapa>_<categoria>_<numero>.png` (ex.: `M01_escrita_01.png`, `M_ALVO_contorno_02.png`).
- Cada linha de `tabela_recortes.csv` aponta para um arquivo que existe em `recortes/`.

## Fora deste prazo

Treino e fine-tuning, data augmentation, mapas de outros cartógrafos, KNN, K-Means, regressão logística e detecção automática (Labelme/YOLO). O código antigo em `recorte_database/` fica no repositório e não faz parte deste piloto.
