# rafa_scripts

Preparação alternativa de mapas para ResNet: extrai a imagem original de cada obra e gera patches 224×224 em RGB, uma pasta por obra.

O pipeline principal do repositório está em [`recorte_database`](../recorte_database/README.md). Este diretório é outro caminho para o mesmo problema.

## Por que esta metodologia

[`recorte_database`](../recorte_database/README.md) renderiza a página, recorta a moldura dourada e aplica letterbox na folha inteira (padrão 512 px, estrutura `ImageFolder`).

Aqui o desenho fica na resolução do arquivo. `extrai_imagens.py` grava os bytes de cada imagem embutida no PDF (`extract_image`). Arquivos que já são imagem são copiados com o mesmo nome. `gera_patches.py` converte para RGB e recorta patches locais de 224×224, o tamanho de entrada da ResNet, sem reduzir a folha inteira para um único quadro.

## Entrada e saída

A origem está fixa em `extrai_imagens.py`: `data/raw/imagens/Albernaz Velho_Exemplos/<obra>`. Cada subpasta é uma obra.

| Etapa | Script | Destino |
|---|---|---|
| Extrair PDF e copiar o resto | `extrai_imagens.py` | `rafa_scripts/output/Albernaz Velho_Exemplos/<obra>` |
| Fatiar patches | `gera_patches.py` | `rafa_scripts/output/patches_albernaz/<obra>` |

De um PDF, cada imagem sai como `{nome do pdf}_pag_XXX_xref_N.ext`, com a extensão original (`jpeg`, `png` e o que o PyMuPDF devolver). O nome do PDF entra no prefixo para dois volumes na mesma pasta não gravarem por cima um do outro. Um `xref` repetido no mesmo PDF é gravado uma vez, na primeira página em que aparece. `.jpg`, `.jpeg`, `.tif` e qualquer outro arquivo da pasta são copiados com o nome original para a mesma pasta de saída.

`rafa_scripts/output/*` está no [`.gitignore`](../.gitignore) da raiz.

## Como rodar

Na raiz do repositório, com o ambiente do [README](../README.md):

```bash
python rafa_scripts/extrai_imagens.py
python rafa_scripts/gera_patches.py
```

A segunda linha imprime, por obra:

```text
[nome da pasta] -> N patches úteis gerados.
```

## Filtros dos patches

`gerar_patches_obra` em `gera_patches.py` usa estes padrões:

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `min_resolucao_img` | 800 | Descarta a imagem se a largura ou a altura for menor que 800 px |
| `tamanho_patch` | 224 | Lado do recorte, em px |
| `stride` | 224 | Passo da janela, sem sobreposição |
| `min_desvio_padrao` | 18.0 | Descarta o patch se o desvio padrão dos pixels for menor que 18 |
| qualidade JPEG | 95 | Arquivo `{nome}_y{yyyy}_x{xxxx}.jpg` |

Extensões lidas: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.jp2`. Cinza, CMYK e RGBA passam por `Image.convert("RGB")` antes do corte. `Image.MAX_IMAGE_PIXELS = None` deixa abrir TIF grandes. Falha num arquivo imprime `Aviso: falha ao processar ...` e o laço segue.

Para calibrar, edite a chamada no final de `gera_patches.py`. Traço fino de pena sumindo: baixe `min_desvio_padrao` para `10` ou `12`. Papel sem desenho ainda passando: suba para `25`. Sobreposição de 50%: `stride=112`.

## Split por obra

Cada obra permanece na própria pasta. Na hora de separar treino e validação, atribua pastas inteiras a um lado ou ao outro. Patches da mesma obra nos dois conjuntos fazem a rede memorizar o tom do papel daquele mapa. Os scripts não criam pastas de treino nem de validação.

## O que este fluxo não faz

- Não há argumentos de linha de comando. Trocar pasta ou limiar é editar `extrai_imagens.py` ou `gera_patches.py`.
- Não detecta moldura dourada e não faz letterbox.
- `get_images` traz selos, ícones e miniaturas junto com os mapas. O corte de 800 px acontece só em `gera_patches.py`.
- A pasta de saída não é esvaziada no início. Uma nova execução sobrescreve arquivos de mesmo nome e deixa o resto.
